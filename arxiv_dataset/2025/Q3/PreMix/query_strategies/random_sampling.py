import random
import torch
import torch.nn as nn

from typing import Optional
from functools import partial

from source.dataset_utils import collate_features
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        super(RandomSampling, self).__init__(model, pool_dataset, train_dataset, collate_fn, batch_size)
    
    def query(self, WSI_budget):
        selected = random.sample(range(len(self.pool_dataset)), WSI_budget)
        return selected