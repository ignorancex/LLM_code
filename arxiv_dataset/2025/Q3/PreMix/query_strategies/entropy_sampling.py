import torch
import torch.nn as nn

from typing import Optional
from functools import partial

from source.dataset_utils import collate_features
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        super(EntropySampling, self).__init__(model, pool_dataset, train_dataset, collate_fn, batch_size)
    
    def query(self, WSI_budget):
        probs, embedings = self.predict_prob_embed(self.model, self.pool_dataset, self.collate_fn, self.batch_size)
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1) # no minus -> from lowest to highest
        selected = U.sort()[1][:WSI_budget].tolist()
        return selected