import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import process_methods as P

# weight maps ----------------------------------------------

def balance_weight_map(gt, epsilon=1e-9):
    axis = tuple(range(2, gt.ndim))
    ccount = gt.sum(axis, keepdim=True)
    c = 1/(torch.sum(1/(epsilon + ccount), axis=1, keepdim=True))
    weight = c / ccount
    weight_map = torch.tile(weight, [1,1] + list(gt.shape[2:]))
    return weight_map

def feedback_weight_map(logits, gt, alpha=3, beta=100):
    probs = F.softmax(logits, dim=1)
    p = torch.sum(probs * gt, axis=1, keepdim=True)
    weight_map = torch.exp(-torch.pow(p, beta)*np.log(alpha))
    weight_map = torch.tile(weight_map, [1] + [logits.shape[1]] + [1]*(logits.ndim-2))
    return weight_map 

# ----------------------------------------------------------


def weighted_cross_entropy(logits, gt, weight):
    prob = F.softmax(logits, dim=1)
    loss = -torch.sum(gt * torch.log(prob) * weight, axis=[1])
    return loss

class CrossEntropy:
    def __init__(self, **kwargs):
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def __call__(self, logits, gt):
        gt = torch.argmax(gt, dim=1)
        loss = self.criterion(logits, gt)
        return loss

class BalanceCrossEntropy:
    def __call__(self, logits, gt):
        weight_map = balance_weight_map(gt)
        loss_map =  weighted_cross_entropy(logits, gt, weight_map)
        loss = loss_map.mean()
        return loss

class FeedbackCrossEntropy:
    def __init__(self, alpha=3, beta=100):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, logits, gt):
        weight_map = feedback_weight_map(logits, gt, self.alpha, self.beta)
        loss_map = weighted_cross_entropy(logits, gt, weight_map)
        loss = loss_map.mean()
        return loss

class SoftDice:
    def __init__(self, ignore_background=True):
        self.ignore_background = ignore_background

    def __call__(self, logits, gt):
        pred = F.softmax(logits, dim=1)
        axis = tuple(range(2, gt.ndim))
        intersection = torch.sum(pred * gt, axis)
        sum_ = torch.sum(pred + gt, axis)
        soft_dice = 2 * intersection / sum_
        if self.ignore_background:
            soft_dice = soft_dice[:, 1:]
        return 1 - soft_dice.mean()

# -------------------regularzation-----------------------------

class L2Norm:
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, parameters):
        reg = 0
        for param in parameters:
            reg += 0.5 * (param ** 2).sum()
        return self.weight * reg
