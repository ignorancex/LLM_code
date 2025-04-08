from __future__ import division
from __future__ import print_function

import math
import os.path as osp
import logging
import random
import sys
import pickle
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.io import list_files
from utils.pcd import KNNSearch, estimate_lrf

log = logging.getLogger(__name__)


def crop_patch(points, kpts, sample_radius, num_points_per_sample):
    """
    Args:
        points (np.array): (N, 3)
        kpts (np.array): (K, 3)
        sample_radius (float):
        num_points_per_sample (int):

    Returns:
        np.array: (K, num_points_per_sample)
    """
    knn_search = KNNSearch(points)
    all_patches = list()
    for i in range(len(kpts)):
        indices = knn_search.query_ball(kpts[i, :], sample_radius)
        if len(indices) <= 1:
            indices = knn_search.query(kpts[i:i + 1, :], num_points_per_sample)
            indices = indices[0, :]
            assert len(indices) == num_points_per_sample
        elif len(indices) > num_points_per_sample:
            indices = np.random.choice(indices, num_points_per_sample, replace=False)
        else:
            indices = np.random.choice(indices, num_points_per_sample, replace=True)
        all_patches.append(indices)
    all_patches = np.asarray(all_patches)
    return all_patches


# ---------------------------------------------------------------------------- #
# Testing dataset
# ---------------------------------------------------------------------------- #
class PointCloudDataset(Dataset):

    def __init__(self,
                 data_root,
                 voxel_size,
                 num_points_per_sample,
                 sample_radius,
                 file_suffix,
                 scale_factor=1.0):
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.num_points_per_sample = num_points_per_sample
        self.sample_radius = sample_radius
        self.file_suffix = file_suffix
        self.scale_factor = scale_factor

        # List data files
        pcd_filenames = list_files(data_root, '*{}'.format(file_suffix), alphanum_sort=True)
        self.pcd_filepaths = [osp.join(data_root, pf) for pf in pcd_filenames]

    def __getitem__(self, index):
        pcd_path = self.pcd_filepaths[index]
        pcd_name = Path(pcd_path).stem

        # Load point cloud
        points = self.load_points(pcd_path)

        # Adaptation
        points = self.rescale(points)
        kpts = points
        kpt_indices = np.arange(len(points))

        # Crop local patches
        patch_indices = crop_patch(points, kpts, self.sample_radius, self.num_points_per_sample)
        shape = (len(kpts), self.num_points_per_sample, 3)
        patches = np.reshape(points[patch_indices.flatten(), :], shape)

        # Estimate LRFs for local patches
        lrfs = [
            estimate_lrf(kpts[k, :], patches[k, :, :].T, self.sample_radius)
            for k in range(len(kpts))
        ]
        lrfs = np.stack(lrfs, axis=0)  # (K, 3, 3)

        ret = {
            'pcd': points,
            'kpts': kpts,
            'kpt_indices': kpt_indices,
            'patches': patches,
            'lrfs': lrfs,
            'name': pcd_name,
        }
        return ret

    def __len__(self):
        return len(self.pcd_filepaths)

    def load_points(self, filepath):
        suffix = Path(filepath).suffix
        if suffix in ('.off', '.obj'):
            # Triangle mesh
            g = o3d.io.read_triangle_mesh(filepath)
            return np.asarray(g.vertices, dtype=np.float32)
        elif suffix in ('.ply',):
            # Point cloud
            g = o3d.io.read_point_cloud(filepath)
            return np.asarray(g.points, dtype=np.float32)
        else:
            raise RuntimeError('Suffix {} is not supported!'.format(suffix))

    def rescale(self, points):
        if self.scale_factor == 1.0:
            return points
        points_temp = np.copy(points)  # (N, 3)
        center = np.mean(points_temp, axis=0, keepdims=True)  # (1, 3)
        points_temp -= center
        points_temp *= self.scale_factor
        return points_temp
