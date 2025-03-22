#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

from itertools import groupby

import numpy as np
import torch
from tqdm import tqdm

__all__ = [
    "density_estimation",
    "density_to_peaks",
    "density_to_peaks_vectorize",
]


def density_estimation(sims, nbrs, labels, **kwargs):
    """use supervised density defined on neigborhood"""
    num, k_knn = sims.shape
    ind_array = labels[nbrs] == np.expand_dims(labels, 1).repeat(k_knn, 1)
    pos = (sims * ind_array).sum(1) # we rely on that where nbr=-1, sim=0.
    neg = (sims * (1 - ind_array)).sum(1)
    conf = np.where(np.all(nbrs==-1, axis=1),
                    0.,
                    (pos - neg))
    num_of_inst = np.where(np.all(nbrs==-1, axis=1),
                           1.,
                           np.sum(nbrs != -1, axis=1))
    conf = conf / num_of_inst
    return conf


def density_to_peaks_vectorize(sims, nbrs, density, max_conn=1, name=""):
    # just calculate 1 connectivity
    assert sims.shape[0] == density.shape[0]
    assert sims.shape == nbrs.shape

    num, k = sims.shape

    if name == "gcn_feat":
        include_mask = nbrs != np.arange(0, num).reshape(-1, 1)
        secondary_mask = (
            np.sum(include_mask, axis=1) == k
        )  # TODO: the condition == k should not happen as simance to the node self should be smallest, check for numerical stability; TODO: make top M instead of only supporting top 1
        include_mask[secondary_mask, -1] = False
        nbrs_exclude_self = nbrs[include_mask].reshape(-1, k - 1)  # (V, 79)
        sims_exclude_self = sims[include_mask].reshape(-1, k - 1)  # (V, 79)
    else:
        include_mask = nbrs != np.arange(0, num).reshape(-1, 1)
        nbrs_exclude_self = nbrs[include_mask].reshape(-1, k - 1)  # (V, 79)
        sims_exclude_self = sims[include_mask].reshape(-1, k - 1)  # (V, 79)

    compare_map = density[nbrs_exclude_self] > density.reshape(-1, 1)
    peak_index = np.argmax(np.where(compare_map, 1, 0), axis=1)  # (V,)
    compare_map_sum = np.sum(compare_map.cpu().data.numpy(), axis=1)  # (V,)

    sim2peak = {
        i: []
        if compare_map_sum[i] == 0
        else [sims_exclude_self[i, peak_index[i]]]
        for i in range(num)
    }
    peaks = {
        i: []
        if compare_map_sum[i] == 0
        else [nbrs_exclude_self[i, peak_index[i]]]
        for i in range(num)
    }

    return sim2peak, peaks


def density_to_peaks(sims, nbrs, density, max_conn=1, sort="sim"):
    # Note that sims has been sorted in descending order
    assert sims.shape[0] == density.shape[0]
    assert sims.shape == nbrs.shape

    num, _ = sims.shape
    sim2peak = {i: [] for i in range(num)}
    peaks = {i: [] for i in range(num)}

    for i, nbr in tqdm(enumerate(nbrs)):
        nbr_conf = density[nbr]
        for j, c in enumerate(nbr_conf):
            nbr_idx = nbr[j]
            if i == nbr_idx or c <= density[i]:
                continue
            sim2peak[i].append(sims[i, j])
            peaks[i].append(nbr_idx)
            if len(sim2peak[i]) >= max_conn:
                break

    return sim2peak, peaks
