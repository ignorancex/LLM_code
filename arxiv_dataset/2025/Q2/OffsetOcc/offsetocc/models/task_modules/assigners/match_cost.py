# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
# adapted from mmdet/models/task_modules/assigners/match_cost.py
from typing import Optional, Union

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost, FocalLossCost
from mmengine.structures import InstanceData
from torch import Tensor

from offsetocc.registry import TASK_UTILS


@TASK_UTILS.register_module()
class L1Cost(BaseMatchCost):
    """L1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 **kwargs) -> Tensor:

        pred_centers3d = pred_instances.center3d
        gt_center3d = gt_instances.center3d

        center3d_cost = torch.cdist(pred_centers3d, gt_center3d, p=1)
        return center3d_cost * self.weight


@TASK_UTILS.register_module()
class L2Cost(BaseMatchCost):
    """L1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 **kwargs) -> Tensor:

        pred_centers3d = pred_instances.center3d
        gt_center3d = gt_instances.center3d

        center3d_cost = torch.cdist(pred_centers3d, gt_center3d, p=2)
        return center3d_cost * self.weight



@TASK_UTILS.register_module()
class MyCrossEntropyLossCost(BaseMatchCost): #TODO: check again the one in mmdet but shapes are incompatible with hungarian assignement
    """CrossEntropyLossCost.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 use_sigmoid: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred: Tensor,
                              gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_queries, 1, *) or
                (num_queries, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_queries, num_gt).
        """
        cls_pred = cls_pred.unsqueeze(1).float()
        gt_labels = gt_labels.unsqueeze(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_masks = pred_instances.scores
        gt_masks = gt_instances.labels
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(pred_masks, gt_masks)
        else:
            raise NotImplementedError

        return cls_cost * self.weight


@TASK_UTILS.register_module()
class MyBinaryFocalLossCost(FocalLossCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries,).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # cls_pred = torch.rand((len(cls_pred), 2))
        # gt_labels = (torch.rand(len(gt_labels)) > 0.5).long()
        n_gt_preds = len(gt_labels)

        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[..., None].expand(-1, n_gt_preds) - neg_cost[..., None].expand(-1, n_gt_preds)
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """

        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels
        return self._focal_loss_cost(pred_scores, gt_labels)