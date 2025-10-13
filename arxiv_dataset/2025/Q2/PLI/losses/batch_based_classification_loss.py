import abc
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class AbstractBaseMetricLoss(nn.Module, ABC):
    @abc.abstractmethod
    def forward(self, ref_features: torch.Tensor, tar_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def code(cls) -> str:
        raise NotImplementedError


class BatchBasedClassificationLoss(AbstractBaseMetricLoss):
    """
    InfoNCE loss can be viewed as a kind of batch-wise classification loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ref_features, tar_features, tau=1.0):
        return self.cal_loss(ref_features, tar_features, tau)

    @classmethod
    def cal_loss(cls, ref_features, tar_features, tau=0.07):
        batch_size = ref_features.size(0)
        device = ref_features.device

        pred = ref_features.mm(tar_features.transpose(0, 1)) / tau
        labels = torch.arange(0, batch_size).long().to(device)
        return F.cross_entropy(pred, labels)

    @classmethod
    def code(cls):
        return 'batch_based_classification_loss'


class SymmetricContrastiveLoss(AbstractBaseMetricLoss):
    """
    Symmetric contrastive loss. Poor performance.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ref_features, tar_features, tau=0.07):
        """
        :param ref_features: (batch_size, feature_dim)
        :param tar_features: (batch_size, feature_dim)
        :param tau: temperature
        :return: loss
        """
        batch_size = ref_features.shape[0]
        device = ref_features.device

        # normalize features
        ref_features = F.normalize(ref_features, dim=1)
        tar_features = F.normalize(tar_features, dim=1)
        logits_per_ref = ref_features.mm(tar_features.transpose(0, 1)) / tau
        logits_per_tar = logits_per_ref.transpose(0, 1)

        labels = torch.arange(0, batch_size).long().to(device)
        total_loss = 0.5 * (F.cross_entropy(logits_per_ref, labels) + F.cross_entropy(logits_per_tar, labels))
        return total_loss

    @classmethod
    def code(cls):
        return 'symmetric_contrastive_loss'