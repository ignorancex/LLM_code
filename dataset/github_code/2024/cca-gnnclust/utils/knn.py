#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""
import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "fast_knns2spmat",
    "mark_same_camera_nbrs",
    "build_knn_per_camera"
]

np.seterr(divide='ignore')

def mark_same_camera_nbrs(knns, cam_ids_oh):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)

    # Set nbr to -1 if from same camera
    same_cam = np.any(cam_ids_oh[:, None, :] & cam_ids_oh[nbrs], axis=-1)
    knns[:, 0, :][same_cam] = -1 # neighbours
    knns[:, 1, :][same_cam] = 0. # similarities

    return knns

def fast_knns2spmat(knns, th_sim=-1):
    # convert knns to symmetric sparse matrix
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)

    nbrs = knns[:, 0, :]
    sims = knns[:, 1, :]

    row, col = np.where(sims >= th_sim)
    # remove the self-loop and same camera instances
    idxs = nbrs[row, col] != -1
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col].astype(int)  # convert to absolute column

    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat

def build_knn_per_camera(features, cam_ids, xws, yws, k=1):
    num_of_nodes = len(cam_ids)
    cam_orders = np.argsort(cam_ids)

    vals, counts = np.unique(cam_ids[cam_orders], return_counts=True)
    num_of_cams = len(counts)

    feat_sorted = features[cam_orders]
    sims = feat_sorted @ feat_sorted.T
    xws_sorted = xws[cam_orders]
    yws_sorted = yws[cam_orders]
    coordinates = np.concatenate((xws_sorted, yws_sorted), axis=1)
    coo_dist = np.linalg.norm(coordinates[:, None, :] - coordinates, axis=-1)
    coo_dist = coo_dist / coo_dist.max()
    coo_dist = coo_dist * (1 - sims)

    cnt_cumsum = np.cumsum(np.concatenate(([0], counts)), axis=0)
    split_edges = np.cumsum(counts)[:-1]

    vert_splits = np.split(sims, split_edges)
    vert_coo_splits = np.split(coo_dist, split_edges)

    # Initialize empty similarities and neighbours matrix
    all_sims = np.empty((num_of_nodes, num_of_cams * k))
    all_sims.fill(0.)
    all_idcs = np.empty((num_of_nodes, num_of_cams * k))
    all_idcs.fill(-1)

    for i, v in enumerate(vert_splits):
        hor_split = np.split(v, split_edges, axis=1)
        hor_coo_split = np.split(vert_coo_splits[i], split_edges, axis=1)
        part_idcs = []
        part_sims = []
        for j, part in enumerate(hor_split):
            if i != j:
                # Check if no enough neighbours
                k = min(part.shape[1], k)
                if part.shape[1] == k:
                    idcs = np.repeat(np.arange(k)[None, :], part.shape[0], axis=0)
                    max_vals = part
                else:
                    k_curr = min(k, part.shape[1] - 1)
                    idcs = np.argpartition(hor_coo_split[j], k_curr)[..., :k_curr]
                    max_vals = np.take_along_axis(part, idcs, 1)

                idcs += cnt_cumsum[j]
                idcs = cam_orders[idcs]
                part_idcs.append(idcs)
                part_sims.append(max_vals)

        if len(part_idcs):
            part_idcs = np.concatenate(part_idcs, axis=1)
            part_sims = np.concatenate(part_sims, axis=1)

            all_idcs[cnt_cumsum[i]:cnt_cumsum[i] + part_idcs.shape[0], :part_idcs.shape[1]] = part_idcs
            all_sims[cnt_cumsum[i]:cnt_cumsum[i] + part_sims.shape[0], :part_sims.shape[1]] = part_sims

    knns = np.concatenate((all_idcs[:, None, :], all_sims[:, None, :]), axis=1)

    remap_fin = ((cam_orders - np.expand_dims(np.arange(len(cam_orders)), 0).T) == 0).nonzero()[1]
    final_knns = knns[remap_fin]

    return final_knns