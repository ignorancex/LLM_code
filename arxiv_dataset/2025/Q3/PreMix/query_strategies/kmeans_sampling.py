import numpy as np
import torch
import torch.nn as nn

from typing import Optional
from functools import partial
from sklearn.cluster import KMeans

from source.dataset_utils import collate_features
from .strategy import Strategy

class KMeansSampling(Strategy):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        super(KMeansSampling, self).__init__(model, pool_dataset, train_dataset, collate_fn, batch_size)
    
    def query(self, WSI_budget):
        embedding = self.get_embedding(self.model, self.pool_dataset, self.collate_fn, self.batch_size)
        embedding = embedding.numpy()
        cluster_learner = KMeans(n_clusters=WSI_budget)
        cluster_learner.fit(embedding)

        cluster_idxs = cluster_learner.predict(embedding)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embedding - centers)**2
        dis = dis.sum(axis=1)
        selected = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(WSI_budget)]).tolist()
        return selected
    
        
