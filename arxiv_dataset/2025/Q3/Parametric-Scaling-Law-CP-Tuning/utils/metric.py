import numpy as np
import torch

from typing import Any
from .utils import Registry

METRICS_REGISTRY = Registry("METRICS")


@METRICS_REGISTRY.register()
def coverage_rate(prediction_sets, labels):
    cvg = 0
    for index, ele in enumerate(zip(prediction_sets, labels)):
        if ele[1] in ele[0]:
            cvg += 1
    return cvg / len(prediction_sets)


@METRICS_REGISTRY.register()
def average_size(prediction_sets, labels):
    avg_size = 0
    for index, ele in enumerate(prediction_sets):
        avg_size += len(ele)
    return avg_size / len(prediction_sets)


@METRICS_REGISTRY.register()
def accuracy(probs, targets, top_k=(1,)):
    k_max = max(top_k)
    batch_size = targets.size(0)

    _, order = probs.topk(k_max, dim=1, largest=True, sorted=True)
    order = order.t()
    correct = order.eq(targets.view(1, -1).expand_as(order))

    acc = []
    for k in top_k:
        correct_k = correct[:k].float().sum()
        acc.append(correct_k.mul_(100.0 / batch_size))
    return acc[0].item()


class Metrics:

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric: {metric} is not defined in DeepCP.")
        return METRICS_REGISTRY.get(metric)


class DimensionError(Exception):
    pass
