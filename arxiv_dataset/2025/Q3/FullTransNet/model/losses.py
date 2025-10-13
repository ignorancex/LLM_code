import torch
from torch.nn import functional as F
import os
import numpy as np
import torch.nn as nn
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def jaccard_loss(true, pred, smooth):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
        eps (int): Smoothing factor
    Returns:
        jacc_loss: the Jaccard loss.
    """
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (torch.sum(true) + torch.sum(pred) - intersection + smooth)
    return (1 - jac) * smooth


def power_jaccard_loss(true, pred, p=1.4, smooth=100):
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (torch.sum(true ** p) + torch.sum(pred ** p) - intersection + smooth)
    return (1 - jac) * smooth


def Tversky_Loss(true, pred, b=0.50, smooth=100):
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (
            torch.sum(true * pred) + b * torch.sum((1 - true) * pred) + (1 - b) * torch.sum(
        true * (1 - pred)) + smooth)
    return (1 - jac) * smooth


def focal_tversky(true, pred, b=0.1, smooth=100):
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (
            torch.sum(true * pred) + b * torch.sum(true * (1 - pred)) + (1 - b) * torch.sum(
        (1 - true) * pred) + smooth)
    pt_1 = jac
    gamma = 2
    return torch.pow((1 - pt_1), gamma) * smooth

# def generate_machine_summ(output, target):
#
#     return pred_summ1,gt_summ

def cul_loss(output, target, loss_name,smooth):


    T = len(target)
    target = torch.as_tensor(target)
    # bos = torch.zeros(( T))
    # eos = torch.zeros(( T))
    # target = torch.cat((bos, target, eos), dim=0)

    # 计算 A 中包含的 1 的位置索引
    indices = torch.nonzero(target).squeeze().cuda()
    # 创建一个全零的二维张量，形状为 (len(indices), len(A))
    target = torch.zeros(len(indices), len(target)).cuda()
    # eos = torch.zeros((1,T)).cuda()
    # target = torch.cat((target, eos), dim=0)
    # 根据索引将 target 中对应的位置置为 1
    for i, index in enumerate(indices):
        target[i][index] = 1
    # zero_tensor = torch.zeros((1, T)).cuda()
    # target = torch.cat((zero_tensor, target), dim=0)


    target_show = target.clone().detach().cpu().numpy()
    gt_summ = target.view(-1)
    pred_summ1 = output.flatten()
    # pred_summ1 = output.view(-1)
    # pred_summ1 = output.reshape(T*len(target))
    # pred_summ1, gt_summ = generate_machine_summ(output, target)

    if loss_name == 'focal':
        loss = calc_cls_loss(pred_summ1, gt_summ, 'focal')
    elif loss_name == 'bce':
        criterion = nn.BCELoss()
        loss = criterion(pred_summ1, gt_summ.cuda())
    elif loss_name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(pred_summ1, gt_summ.cuda())
    elif loss_name == 'jaccard':
        loss = jaccard_loss(gt_summ, pred_summ1,smooth)
    elif loss_name == 'focal_tversky':
        loss = focal_tversky(gt_summ, pred_summ1,smooth)
    elif loss_name == 'power_jaccard':
        loss = power_jaccard_loss(gt_summ, pred_summ1)
    elif loss_name == 'tversky':
        loss = Tversky_Loss(gt_summ, pred_summ1)

    return loss


def calc_cls_loss(pred: torch.Tensor,
                  test: torch.Tensor,
                  kind: str = 'focal'
                  ) -> torch.Tensor:
    """Compute classification loss on both positive and negative samples.

    :param pred: Predicted class. Sized [N, S].
    :param test: Class label where 1 marks positive, -1 marks negative, and 0
        marks ignored. Sized [N, S].
    :param kind: Loss type. Choose from (focal, cross-entropy).
    :return: Scalar loss value.
    """
    test = test.type(torch.long)
    num_pos = test.sum()
    # print(test)
    pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)

    if kind == 'focal':
        loss = focal_loss(pred, test, reduction='sum')
    elif kind == 'cross-entropy':
        loss = F.nll_loss(pred.log(), test)
    else:
        raise ValueError(f'Invalid loss type {kind}')

    # print(loss,num_pos)
    loss = loss / num_pos
    return loss







def calc_ctr_loss(pred, test, pos_mask):
    pos_mask = pos_mask.type(torch.bool)

    pred = pred[pos_mask]
    test = test[pos_mask]
    loss = F.binary_cross_entropy(pred, test)
    # try:
    #
    # except Exception as e:
    #     pass
    return loss


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form.

    :param labels: Class labels. Sized [N].
    :param num_classes: Number of classes.
    :return: One-hot encoded labels. sized [N, #classes].
    """
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def focal_loss(x: torch.Tensor,
               y: torch.Tensor,
               alpha: float = 0.25,
            #    alpha: float = 0.1,
               gamma: float = 2,
               reduction: str = 'sum'
               ) -> torch.Tensor:
    """Compute focal loss for binary classification.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    :param x: Predicted confidence. Sized [N, D].
    :param y: Ground truth label. Sized [N].
    :param alpha: Alpha parameter in focal loss.
    :param gamma: Gamma parameter in focal loss.
    :param reduction: Aggregation type. Choose from (sum, mean, none).
    :return: Scalar loss value.
    """
    _, num_classes = x.shape

    t = one_hot_embedding(y, num_classes)
    t = t.cuda()
    # p_t = p if t > 0 else 1-p
    p_t = x * t + (1 - x) * (1 - t)
    # alpha_t = alpha if t > 0 else 1-alpha
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()

    if reduction == 'sum':
        fl = fl.sum()
    elif reduction == 'mean':
        fl = fl.mean()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Invalid reduction mode {reduction}')

    return fl


def focal_loss_with_logits(x, y, reduction='sum'):
    """Compute focal loss with logits input"""
    return focal_loss(x.sigmoid(), y, reduction=reduction)


