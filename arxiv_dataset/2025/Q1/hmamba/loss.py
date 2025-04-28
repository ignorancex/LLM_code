# -*- coding: utf-8 -*-
# @Time    : 05/15/24
# @Author  : Fu-An Chao
# @Affiliation  : National Taiwan Normal University
# @Email   : fuann@ntnu.edu.tw
# @File    : loss.py

# code modified from https://github.com/hirofumi0810/neural_sp

import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_lsm(logits, ys, lsm_prob, ignore_index, training,
                      normalize_length=False):
    """Compute cross entropy loss for label smoothing of sequence-to-sequence models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`
        lsm_prob (float): label smoothing probability
        ignore_index (int): index for padding
        normalize_length (bool): normalize XE loss by target sequence length
    Returns:
        loss (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()
    ys = ys.view(-1)
    logits = logits.view((-1, vocab))  # `[B * T, vocab]`

    if lsm_prob == 0 or not training:
        loss = F.cross_entropy(logits, ys,
                               ignore_index=ignore_index, reduction='mean')
        if not normalize_length:
            loss *= (ys != ignore_index).sum() / float(bs)
    else:
        with torch.no_grad():
            target_dist = logits.new_zeros(logits.size())
            target_dist.fill_(lsm_prob / (vocab - 1))
            mask = (ys == ignore_index)
            ys_masked = ys.masked_fill(mask, 0)
            target_dist.scatter_(1, ys_masked.unsqueeze(1), 1 - lsm_prob)  # `[B * T, vocab]`

        log_probs = torch.log_softmax(logits, dim=-1)
        loss_sum = -torch.mul(target_dist, log_probs)
        n_tokens = len(ys) - mask.sum().item()
        denom = n_tokens if normalize_length else bs
        loss = loss_sum.masked_fill(mask.unsqueeze(1), 0).sum() / denom


    return loss


def decoupled_cross_entropy_lsm(
    logits, realphns, canophns, lsm_prob_m=0.1, lsm_prob_c=0.0, a=0.70, ignore_index=-1, training=True):
    """Compute decoupled cross entropy loss for MDD task.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        realphns (FloatTensor): `[B, L]`
        canophns (FloatTensor): `[B, L]`
        lsm_prob_m (float): label smoothing probability for mis phns
        lsm_prob_c (float): label smoothing probability for cor phns
        a (float): weight magnitude for misp
        b (float): weight for dexent loss
        ignore_index (int): index for padding
    Returns:
        loss (FloatTensor): `[1]`

    """
    pad_mask = (realphns>=0)
    cor_mask = (canophns == realphns)
    mis_mask = (canophns != realphns)
    cor_realphns = realphns.masked_fill(mis_mask, ignore_index) # mask mis pos with -1
    mis_realphns = realphns.masked_fill(cor_mask, ignore_index)  # mask cor pos with -1
    num_cor = (cor_mask).sum()
    num_mis = (mis_mask).sum()
    w_mis = (num_cor / num_mis) ** a

    loss_cor = cross_entropy_lsm(logits, cor_realphns.long(), lsm_prob=lsm_prob_c, ignore_index=ignore_index, training=training)
    loss_mis = cross_entropy_lsm(logits, mis_realphns.long(), lsm_prob=lsm_prob_m, ignore_index=ignore_index, training=training)
    loss = loss_cor + w_mis * loss_mis

    return loss
