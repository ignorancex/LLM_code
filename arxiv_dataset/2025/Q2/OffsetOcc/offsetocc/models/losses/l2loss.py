# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
# from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/smooth_l1_loss.py
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from offsetocc.registry import MODELS
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def l2_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    # loss = torch.abs(pred - target)
    loss = torch.norm(pred - target, p=2, dim=-1)
    # margin mx = 0.005, my = 0.005, mz = 0.0625
    # m = torch.tensor([[0.005, 0.005, 0.0625]], device=pred.device)
    # diff = torch.max(torch.abs(pred - target) - m, torch.zeros_like(pred))
    # loss = torch.norm(diff, p=2, dim=-1)
    # ############ ----- loss = torch.max(loss - m, torch.zeros_like(loss)) --> loss with margin ------ not sure
    return loss

@MODELS.register_module()
class L2Loss(nn.Module):
    """L2 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_l2 = self.loss_weight * l2_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_l2

@weighted_loss
def l2margin_loss(pred: Tensor, target: Tensor, margin: list) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    margin = torch.tensor(margin, device=pred.device)

    assert pred.size() == target.size()
    diff = torch.max(torch.abs(pred - target) - margin, torch.zeros_like(pred))
    loss = torch.norm(diff, p=2, dim=-1)
    return loss

@MODELS.register_module()
class L2MarginLoss(nn.Module):
    """L2 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 margin: list = [0.005, 0.005, 0.0625],
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_l2 = self.loss_weight * l2margin_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, margin=self.margin)
        return loss_l2
