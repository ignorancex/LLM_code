import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import tqdm
import pickle
import random
import torch.utils.data as data

class GoldenDataset(data.Dataset):
    def __init__(self, db_k, db_ids, db_id_to_embed_paths):
        # param db_k: num views per scene in dataset
        # param db_ids: list of string ids corresponding to images
        # param db_id_to_embed_paths: dict from string ids to absolute paths
        # where each torch tensor embedding is stored (npz)
        # TODO support db_k = None, i.e. infer from db_ids
        self.db_k = db_k
        self.db_ids = db_ids
        self.db_id_to_embed_paths = db_id_to_embed_paths
        self.positive_k = 2 # we use positive pairs

    def __len__(self):
        return len(self.db_ids)

    def __getitem__(self, idx):
        # param idx: int
        # returns a list of 2 embeddings representing 2 random embedded views
        # of the idx-th scene

        idxs = random.sample(range(self.db_k), self.positive_k)
        all_embed_paths = np.array(self.db_id_to_embed_paths[self.db_ids[idx]]) # temporarily to np array
        embed_paths = all_embed_paths[idxs].tolist() # back to list
        # dummy label list otherwise it will ignore the second element of my list thinking it's a label
        return [torch.from_numpy(np.load(embed_path)) for embed_path in embed_paths], []