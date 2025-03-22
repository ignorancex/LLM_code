import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from eunet.core import multi_apply


def cmp_error(pred, label, cmp_key=None, hop_mask=None, weight=None, reduction='sum', avg_factor=None, min_diff=1e-8, eps=1e-7, scalar=1.0, **kwargs):
    """Pred should larger than label

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    eps = min(min_diff, eps)
    delta_pred = pred - label - min_diff
    if cmp_key is not None:
        delta_pred *= torch.sign(cmp_key)
    if hop_mask is not None:
        delta_pred *= hop_mask
    mask = delta_pred < 0
    mask.requires_grad = False

    # element-wise losses
    invalid_pred = delta_pred * mask
    loss = (invalid_pred*scalar) ** 2
    loss = torch.sum(loss, dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    avg_factor = torch.sum(mask) + eps if reduction == 'mean' else None
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(loss)

    return loss,


@LOSSES.register_module()
class CmpLoss(nn.Module):
    """Cross entropy loss

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_lovasz'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 min_diff=1e-8,
                 scalar=1.0,
                 loss_name='loss_cmp',):
        super(CmpLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.min_diff = min_diff
        self.scalar = scalar

        self.criterion = cmp_error
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                cmp_key=None,
                hop_mask=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        '''
            pred should be larger than gt_label
        '''
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        bs = len(label)
        loss = multi_apply(
            self.criterion,
            cls_score, label, cmp_key if cmp_key is not None else [None]*bs,
            hop_mask if hop_mask is not None else [None]*bs,
            weight=weight, reduction=reduction, avg_factor=avg_factor, min_diff=self.min_diff, scalar=self.scalar, **kwargs)[0]
        loss = torch.stack(loss).mean()
        loss *= self.loss_weight
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
