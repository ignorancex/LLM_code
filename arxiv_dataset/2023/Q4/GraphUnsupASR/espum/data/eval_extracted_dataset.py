# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset, Dictionary, data_utils
import faiss

logger = logging.getLogger(__name__)


class EvalExtractedDataset(FairseqDataset):
    def __init__(
        self,
        path,
        split,
        dict_path,
        centroid_path,
        min_length=3,
        max_length=None,
        labels='phn',
        shuffle=True,
        sort_by_length=True,
        aux_target_postfix=None,
        segment_path=None,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        with open(dict_path, 'r'):
            self.label_dict = Dictionary.load(dict_path)
        
        self.sizes = []
        self.offsets = []
        self.labels = []

        path = os.path.join(path, split)
        data_path = path
        self.data = np.load(path + ".npy")
        offset = 0
        if not os.path.exists(path + f".{labels}"):
            labels = None

        with open(path + ".lengths", "r") as len_f, open(path + f".{labels}", "r") as lbl_f:
            for line in len_f:
                length = int(line.rstrip())
                lbl = next(lbl_f).rstrip().split()
                if length >= min_length and (
                    max_length is None or length <= max_length
                ):
                    self.sizes.append(length)
                    self.offsets.append(offset)
                    if lbl is not None:
                        self.labels.append(lbl)
                offset += length

        centroids = np.load(centroid_path)
        self.n_clus, d = centroids.shape
        index_flat = (
            faiss.IndexFlatL2(d)
        )
        self.faiss_index = index_flat
        self.faiss_index.add(centroids)
        
        self.sizes = np.asarray(self.sizes)
        self.offsets = np.asarray(self.offsets)

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.data[offset:end].copy()).float()
        z = torch.from_numpy(self.faiss_index.search(feats, 1)[1]).squeeze(-1)
        clus_feats = F.one_hot(z, self.n_clus).float()
        
        res = {"id": index, "features": feats, "clus_features": clus_feats}
        if len(self.labels):
            res["target"] = self.label_dict.encode_line(
                self.labels[index],
                line_tokenizer=lambda x: x,
                append_eos=False,
            )

        return res

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        clus_features = [s["clus_features"] for s in samples]
        sizes = [len(s) for s in features]
 
        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        collated_clus_features = None
        if clus_features[0] is not None:
            collated_clus_features = clus_features[0].new_zeros(
                len(clus_features), target_size, clus_features[0].size(-1)
            )

        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
 
        for i, (f, cf, size) in enumerate(zip(features, clus_features, sizes)):
            collated_features[i, :size] = f
            if cf is not None:
                collated_clus_features[i, :size] = cf
            padding_mask[i, size:] = True
            res = {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": {
                    "features": collated_features,
                    "clus_features": collated_clus_features,
                    "padding_mask": padding_mask,
                }
            }

            if len(self.labels) > 0:
                target = data_utils.collate_tokens(
                    [s["target"] for s in samples],
                    pad_idx=self.label_dict.pad(),
                    left_pad=False,
                )
                res["target"] = target
        
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]

    def __len__(self):
        return len(self.sizes)
