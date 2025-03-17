import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1, size_average=False):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    loss = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
    if size_average:
        loss = loss.mean()
    return loss
