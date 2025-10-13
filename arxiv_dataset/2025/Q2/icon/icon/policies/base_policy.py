import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from torch.optim import Optimizer
from icon.utils.normalizer import Normalizer


class BasePolicy(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.normalizer = Normalizer()

    def set_normalizer(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def compute_loss(self, batch: Dict) -> Tensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def predict_action(self, obs: Dict) -> Dict:
        raise NotImplementedError
    
    def get_optimizer(self, *args, **kwargs) -> Optimizer:
        raise NotImplementedError
    
    def state_dicts(self) -> Dict:
        state_dicts = dict(
            policy=self.state_dict(),
            normalizer=self.normalizer
        )
        return state_dicts

    def load_state_dicts(self, state_dicts: Dict) -> None:
        self.load_state_dict(state_dicts['policy'])
        self.normalizer = state_dicts['normalizer']