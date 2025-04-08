import os.path
import torch
import warnings

import copy
import timm
import argparse
import numpy as np
from pathlib import Path
import torch.nn as nn
from monai.metrics import compute_hausdorff_distance
from transformers import ViTForImageClassification, ViTModel
from torch.nn import MarginRankingLoss


def compute_rank_loss(args, output_dice, true_dices, criterionRank):
    b = output_dice.shape[0]
    rank_loss = 0
    for i in range(b):
        for j in range(b):
            if i != j:
                if true_dices[i] >= true_dices[j]:
                    rank_loss += criterionRank(output_dice[i].reshape(1), output_dice[j].reshape(1),
                                               torch.tensor(1.0, device=args.local_rank).reshape(1)).cuda(args.local_rank)
                else:
                    rank_loss += criterionRank(output_dice[i].reshape(1), output_dice[j].reshape(1),
                                               torch.tensor(-1.0, device=args.local_rank).reshape(1)).cuda(args.local_rank)
    return rank_loss


def calc_HD(crop_predict, crop_mask):
    crop_predict = np.expand_dims(np.expand_dims(crop_predict, axis=0), axis=0)
    crop_mask = np.expand_dims(np.expand_dims(crop_mask, axis=0), axis=0)

    gt_hd = compute_hausdorff_distance(crop_predict, crop_mask, include_background=False,
                                       distance_metric='euclidean', percentile=None, directed=False, spacing=None)
    return gt_hd.item()


def calc_dice(pred, gt):
    pred = np.reshape(pred, newshape=(1, -1))
    gt = np.reshape(gt, newshape=(1, -1))
    smooth = 1
    intersection = (pred*gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)