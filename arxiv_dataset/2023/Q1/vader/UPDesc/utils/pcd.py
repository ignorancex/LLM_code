from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
import open3d as o3d
import torch
import math
import os
import os.path as osp
import sys
from scipy.spatial import cKDTree


class KNNSearch(object):

    def __init__(self, points):
        # points: (N, 3)
        self.points = np.asarray(points, dtype=np.float32)
        self.kdtree = cKDTree(points)

    def query(self, kpts, num_samples):
        # kpts: (K, 3)
        kpts = np.asarray(kpts, dtype=np.float32)
        nndists, nnindices = self.kdtree.query(kpts, k=num_samples, n_jobs=-1)
        assert len(kpts) == len(nnindices)
        return nnindices  # (K, num_samples)

    def query_ball(self, kpt, radius):
        # kpt: (3, )
        kpt = np.asarray(kpt, dtype=np.float32)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, n_jobs=-1)  # list
        return nnindices


def estimate_lrf(pt, ptnn, patch_kernel):
    """Re-implementation (+adaptation) of the LRF computed in:
    Z. Gojcic, C. Zhou, J. Wegner, and W. Andreas,
    "The perfect match: 3D point cloud matching with smoothed densities"
    CVPR, 2019
    https://github.com/fabiopoiesi/dip/blob/master/lrf.py

    Args:
        pt (np.array): (3, )
        ptnn (np.array): (3, NN-1), without pt
        patch_kernel (float):

    Returns:
        np.array: (3, 3)
    """
    ptnn_pt = ptnn - pt[:, np.newaxis]  # (3, NN-1)

    # eq. 3
    ptnn_cov = (1.0 / ptnn.shape[-1]) * np.dot(ptnn_pt, ptnn_pt.T)  # (3, 3)

    # The normalized (unit "length") eigenvectors, s.t. the column eigvecs[:,i] is
    # the eigenvector corresponding to the eigenvalue eigvals[i].
    eigvals, eigvecs = np.linalg.eig(ptnn_cov)
    smallest_eigval_idx = np.argmin(eigvals)
    np_hat = eigvecs[:, smallest_eigval_idx]  # (3, )

    # eq. 4
    zp = np_hat if np.sum(np.dot(np_hat, -ptnn_pt)) >= 0 else -np_hat  # (3, )
    zp /= np.linalg.norm(zp)

    ptnn_pt_zp = np.dot(ptnn_pt.T, zp[:, np.newaxis])  # (NN-1, 1)
    v = ptnn_pt - (ptnn_pt_zp * zp).T  # (3, NN-1)
    # eq. 6
    # (NN-1, )
    alpha = (patch_kernel - np.linalg.norm(-ptnn_pt, axis=0))**2
    # (NN-1, )
    beta = ptnn_pt_zp.squeeze()**2

    # eq. 5
    v_alpha_beta = np.dot(v, (alpha * beta)[:, np.newaxis])  # (3, 1)
    if np.linalg.norm(v_alpha_beta) < 1e-4:
        xp = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        xp = v_alpha_beta / np.linalg.norm(v_alpha_beta)
        xp = xp.squeeze()  # (3, 1) -> (3, )

    yp = np.cross(zp, xp)  # (3, )
    yp /= np.linalg.norm(yp)

    xp = np.cross(yp, zp)  # (3, )
    xp /= np.linalg.norm(xp)

    # LRF
    lRg = np.stack((xp, yp, zp), axis=1)  # Each col is an axis
    return np.asarray(lRg, dtype=np.float32)


def filter_outliers(points, normals=None, nb_points=256, radius=0.3):
    dtype = points.dtype
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_flt, _ = pcd.remove_radius_outlier(nb_points, radius)
    out_points = np.asarray(pcd_flt.points, dtype=dtype)
    if normals is not None:
        out_normals = np.asarray(pcd_flt.normals, dtype=dtype)
    else:
        out_normals = None
    return out_points, out_normals


def farthest_point_sampling(points, max_points=512, gpu=-1):
    # Ref:
    # https://github.com/pytorch/pytorch/issues/40403#issuecomment-648515174
    import torch_cluster

    if gpu >= 0:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    npoints = len(points)
    indices = torch_cluster.fps(torch.as_tensor(points, dtype=torch.float32, device=device),
                                torch.zeros(npoints, dtype=torch.int64, device=device),
                                ratio=float(max_points) / npoints,
                                random_start=True)
    indices = indices.cpu().numpy()
    return indices
