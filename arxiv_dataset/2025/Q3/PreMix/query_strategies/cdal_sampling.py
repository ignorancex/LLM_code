import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from functools import partial
from sklearn.metrics import pairwise_distances

from source.dataset_utils import collate_features
from .strategy import Strategy


import numpy as np

class CDALSampling(Strategy):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        super(CDALSampling, self).__init__(model, pool_dataset, train_dataset, collate_fn, batch_size)

    def select_coreset(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        print('selecting coreset...')
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, WSI_budget):

        probs, embeddings = self.predict_prob_embed(self.model, self.pool_dataset, self.collate_fn, self.batch_size)
        probs_l, _ = self.predict_prob_embed(self.model, self.train_dataset, self.collate_fn, self.batch_size)

        selected = self.select_coreset(F.softmax(probs, dim=1).numpy(), F.softmax(probs_l, dim=1).numpy(), WSI_budget)
        return selected
