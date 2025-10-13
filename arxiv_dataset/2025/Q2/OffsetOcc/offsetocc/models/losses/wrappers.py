"""Module with registrable loss wrappers."""

# Python Libraries
import torch
from torch import nn

# Local Modules
from offsetocc.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy loss wrapper."""

    def __init__(
            self,
            weight: list[float] = None,
            ignore_index: int = 255,
            **kwargs
    ):
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        super().__init__(weight, ignore_index=ignore_index, **kwargs)
