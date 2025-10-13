import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Union, Dict


def to(x: Union[Dict, Optimizer], device: torch.device) -> Union[Dict, Optimizer]:
    """
    Similar to pytorch to(device) function, but can be
    applied to dictionaries and pytorch optimizers
    """
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict):
                x[k] = to(v, device)
            elif isinstance(v, Tensor):
                x[k] = v.to(device)
    elif isinstance(x, Optimizer):
        for state in x.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(device)
    return x