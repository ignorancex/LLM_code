"""
Image statistics utilities.
"""

import torch
from torch import nn


def w_sum(imgs, mask, rng=None):  # pylint: disable=W0613
    return torch.sum(imgs * mask.unsqueeze(dim=0), dim=(1, 2)).unsqueeze(dim=1)


def max_mean_bin(imgs, masks, rng=None):  # pylint: disable=W0613
    means = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device
    )
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()
        )
        means[:, i] += torch.mean(imgs_slct, dim=1)
    return torch.max(means, dim=1)[0].unsqueeze(dim=1)


# pylint: disable=W0613
def argmax_mean_bin(imgs, masks, rng=None, beta=None):
    means = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device
    )
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()
        )
        means[:, i] += torch.mean(imgs_slct, dim=1)
    if beta is None:
        id_ = torch.argmax(means, dim=1)
        return nn.functional.one_hot(  # pylint: disable=E1102
            id_.to(torch.int64), num_classes=len(masks)
        )
    return nn.functional.softmax(beta * means, dim=1)


def max_std_bin(imgs, masks, rng=None):  # pylint: disable=W0613
    stds = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device
    )
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()
        )
        stds[:, i] += torch.std(imgs_slct, dim=1)
    return torch.max(stds, dim=1)[0].unsqueeze(dim=1)


# pylint: disable=W0613
def argmax_std_bin(imgs, masks, rng=None, beta=None):
    stds = torch.zeros(
        imgs.size(0), len(masks), dtype=imgs.dtype, device=imgs.device
    )
    for i, mask in enumerate(masks):
        imgs_slct = imgs.flatten(start_dim=1).index_select(
            dim=1, index=mask.flatten().nonzero().squeeze()
        )
        stds[:, i] += torch.std(imgs_slct, dim=1)
    if beta is None:
        id_ = torch.argmax(stds, dim=1)
        return nn.functional.one_hot(  # pylint: disable=E1102
            id_.to(torch.int64), num_classes=len(masks)
        )
    return nn.functional.softmax(beta * stds, dim=1)


# pylint: disable=E1102
def _similarity(a, b):
    a = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
    b = b / torch.linalg.vector_norm(b, dim=1, keepdim=True)
    return torch.linalg.vecdot(a, b, dim=1)


def max_sim_bin(imgs, masks, rng=None):  # pylint: disable=W0613
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(
            imgs.flatten(start_dim=1).index_select(
                dim=1, index=mask.flatten().nonzero().squeeze()
            )
        )
    sims = torch.zeros(
        imgs.size(0), len(masks) - 1, dtype=imgs.dtype, device=imgs.device
    )
    for i, img_slct in enumerate(imgs_slct[1:]):
        sims[:, i] += _similarity(imgs_slct[0], img_slct)
    return torch.max(sims, dim=1)[0].unsqueeze(dim=1)


# pylint: disable=W0613
def argmax_sim_bin(imgs, masks, rng=None, beta=None):
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(
            imgs.flatten(start_dim=1).index_select(
                dim=1, index=mask.flatten().nonzero().squeeze()
            )
        )
    sims = torch.zeros(
        imgs.size(0), len(masks) - 1, dtype=imgs.dtype, device=imgs.device
    )
    for i, imgs_slct_i in enumerate(imgs_slct[1:]):
        sims[:, i] += _similarity(imgs_slct[0], imgs_slct_i)
    if beta is None:
        id_ = torch.argmax(sims, dim=1)
        return nn.functional.one_hot(  # pylint: disable=E1102
            id_.to(torch.int64), num_classes=len(masks)
        )
    return nn.functional.softmax(beta * sims, dim=1)


def max_sim_rand00_bin(imgs, masks, rng, probs=None):
    n_sims = len(masks) // 2
    if probs is not None:
        assert len(probs) == n_sims
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(
            imgs.flatten(start_dim=1).index_select(
                dim=1, index=mask.flatten().nonzero().squeeze()
            )
        )
    sims = torch.zeros(
        imgs.size(0), n_sims, dtype=imgs.dtype, device=imgs.device
    )
    for i in range(n_sims):
        if (probs is None) or torch.bernoulli(
            torch.full((1,), 1.0 - probs[i]), generator=rng
        ):
            sims[:, i] += _similarity(imgs_slct[i], imgs_slct[n_sims + i])
    return torch.max(sims, dim=1)[0].unsqueeze(dim=1)


def argmax_sim_rand01_bin(imgs, masks, rng, permute=False, beta=None):
    n_patch = len(masks) // 2
    # Remove error masks from main masks
    masks = masks.clone()
    for i in range(n_patch):
        masks[i] -= masks[i + n_patch]
    # Collect pixels
    imgs_slct = []
    for mask in masks:
        imgs_slct.append(
            imgs.flatten(start_dim=1).index_select(
                dim=1, index=mask.flatten().nonzero().squeeze()
            )
        )
    # Apply permutation on error masks
    if permute:
        perm = torch.randperm(n_patch - 1, generator=rng) + n_patch + 1
    else:
        perm = torch.arange(n_patch - 1) + n_patch + 1
    # Compute similarities
    sims = torch.zeros(
        imgs.size(0), n_patch - 1, dtype=imgs.dtype, device=imgs.device
    )
    ref = torch.concatenate((imgs_slct[0], imgs_slct[n_patch]), dim=1)
    for i in range(n_patch - 1):
        sims[:, i] += _similarity(
            ref,
            torch.concatenate((imgs_slct[i + 1], imgs_slct[perm[i]]), dim=1),
        )
    if beta is None:
        id_ = torch.argmax(sims, dim=1)
        return nn.functional.one_hot(  # pylint: disable=E1102
            id_.to(torch.int64), num_classes=n_patch - 1
        )
    return nn.functional.softmax(beta * sims, dim=1)
