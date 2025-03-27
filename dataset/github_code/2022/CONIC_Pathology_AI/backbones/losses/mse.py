from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ._functional import soft_dice_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["DiceLoss"]


class MSELoss(_Loss):

    def __init__(
        self,
    ):
        """Implementation of MSE loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        """
        super(MSELoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = y_pred - y_true
        loss = (loss * loss).mean()
        return loss
