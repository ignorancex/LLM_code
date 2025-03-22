from collections import OrderedDict
from typing import List, Optional, Sequence

import numpy as np
import torch
from prettytable import PrettyTable


class Avg_values():
    def __init__(self):
        self.total = None
        self.count = None
    
    def update(self, value, n):
        if self.total is None or self.count is None:
            self.total = value * n
            self.count = n
        else:
            self.total += value * n
            self.count += n
    
    @property
    def avg(self):
        return self.total / self.count
        

def to_tensor(value):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


def cls_accuracy(pred, target, topk=(1,)):
    """classification Tsak Metric: Accuracy

    Args:
        cls_score (_type_): N * d
        target (_type_): N or N*d
        topk (tuple, optional): Tuple(int). Defaults to (1,).
    """
    pred = to_tensor(pred)
    target = to_tensor(target).to(torch.int64)
    num = pred.size(0)
    assert pred.size(0) == target.size(0), \
        f"The size of pred ({pred.size(0)}) doesn't match "\
        f'the target ({target.size(0)}).'  
    
    # For pred score, calculate on all topk and thresholds.
    pred = pred.float()
    maxk = max(topk)

    if maxk > pred.size(1):
        raise ValueError(
            f'Top-{maxk} accuracy is unavailable since the number of '
            f'categories is {pred.size(1)}.')

    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    results = dict()
    for k in topk:
        _correct = correct
        correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100. / num)
        results[f'top{k}'] = acc
    return results


def seg_metrics(data_samples, num_classes, ignore_index):
    results = []
    for data_sample in data_samples:
        pred_label = data_sample['pred_sem_seg']['data'].squeeze()

        label = data_sample['gt_sem_seg']['data'].squeeze().to(
            pred_label)
        results.append(intersect_and_union(pred_label, label, num_classes, ignore_index))

    return results
            

def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
