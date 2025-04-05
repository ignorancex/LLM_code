import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_dice_loss(logits, targets, smooth=1.0): # targets is one hot
    probs = logits.softmax(dim=1)
    n_classes = logits.shape[1]
    loss = 0
    for i_class in range(n_classes):
        if targets[:,i_class].sum()>0:
            loss += dice_loss_perclass(probs[:,i_class], targets[:,i_class], smooth)
    return loss / n_classes

def dice_loss_perclass(probs, targets, smooth=1.):
    intersection = probs * targets.float()
    # print(intersection.sum().item(), probs.sum().item()+targets.sum().item())
    if 1 - (2. * intersection.sum()+smooth) / (probs.sum()+targets.sum()+smooth)<0:
        print(intersection.sum().item(), probs.sum().item()+targets.sum().item())
    return 1 - (2. * intersection.sum()+smooth) / (probs.sum()+targets.sum()+smooth)


def soft_cross_entropy_loss(pred_logit, target):
    log_pred = F.log_softmax(pred_logit, dim=-1)
    loss = -(log_pred * target).mean()
    return loss
