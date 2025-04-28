import math
import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    
    C = tensor.size(1)        # image shape
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))     
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)              


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        intersect = (output * target).sum(-1).sum() + self.epsilon
        denominator = ((output + target).sum(-1)).sum() + self.epsilon

        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class EMLLoss(nn.Module):
    def __init__(self):
        super(EMLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Code written by Seung hyun Hwang
        gamma = 1.1
        alpha = 0.48
        smooth = 1.
        epsilon = 1e-7
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        # dice loss
        intersection = (y_true * y_pred).sum()
        dice_loss = (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

        # focal loss
        y_pred = torch.clamp(y_pred, epsilon)

        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
        focal_loss = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - \
                     torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
        return focal_loss - torch.log(dice_loss)
