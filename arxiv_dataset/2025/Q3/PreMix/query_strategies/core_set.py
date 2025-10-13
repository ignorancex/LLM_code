import numpy as np
import torch
import torch.nn as nn

from typing import Optional
from functools import partial
from sklearn.metrics import pairwise_distances
from datetime import datetime

from source.dataset_utils import collate_features
from .strategy import Strategy

class Coreset(Strategy):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        super(Coreset, self).__init__(model, pool_dataset, train_dataset, collate_fn, batch_size)
        self.tor = 1e-4
    
    def further_first(self, X, X_set, WSI_budget):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)
        
        idxs = []

        for i in range(WSI_budget):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        
        return idxs
    
    def query(self, WSI_budget):
        t_start = datetime.now()
        embed = self.get_embedding(self.model, self.pool_dataset, self.collate_fn, self.batch_size)
        embed_train = self.get_embedding_train(self.model, self.train_dataset, self.collate_fn, self.batch_size)
        embedding = embed.numpy()
        embedding_train = embed_train.numpy()

        selected = self.further_first(embedding, embedding_train, WSI_budget)

        return selected
        
