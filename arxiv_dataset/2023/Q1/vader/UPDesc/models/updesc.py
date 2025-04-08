from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import sys
import logging
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from diffvoxel.voxelization import voxelize, create_voxel_grids, transform_voxel_grids

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models import pointnet
from models.transformer import Transformer
from models.cnn3d import cnn3ds
from models.pcpnet import PCPNet

log = logging.getLogger(__name__)


def is_invalid_tensor(x):
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


class UPDesc(nn.Module):

    def __init__(self, **hparams):
        super().__init__()

        # Configs
        self.hparams = hparams

        # Desc extractor
        self.edge_length = hparams['voxelization.edge_length']
        self.resolution = hparams['voxelization.resolution']
        self.sigma = hparams['voxelization.sigma']

        # Constant variables
        voxel_size = self.edge_length / self.resolution

        voxel_grids = create_voxel_grids(self.edge_length, self.resolution)
        voxel_grids = np.reshape(voxel_grids, (-1, 3))
        self.register_buffer('voxel_grids', torch.from_numpy(voxel_grids))  # (V, 3)

        voxel_radius = voxel_size / 2.0
        voxel_radii = np.ones((len(voxel_grids),), dtype=np.float32) * voxel_radius
        self.register_buffer('voxel_radii', torch.from_numpy(voxel_radii))  # (V, )

        # Model sub-modules
        self.local_pointnet = pointnet.get_encoder(
            hparams['local_pointnet.type'], hparams['data.type'])(
                in_channels=hparams['local_pointnet.in_channels'],
                out_channels=hparams['local_pointnet.out_channels'],
                use_bn=hparams['local_pointnet.use_bn'],
                use_xyz=hparams['local_pointnet.use_xyz'],
            )

        self.transformer = Transformer(
            in_channels=hparams['local_pointnet.out_channels'],
            num_rotations=hparams['transformer.num_rotations'],
            dim_rotations=hparams['transformer.dim_rotations'],
            num_scales=hparams['transformer.num_scales'],
            max_scale=hparams['transformer.max_scale'],
            min_scale=hparams['transformer.min_scale'],
            dropout=hparams['transformer.dropout'],
            dropout_prob=hparams['transformer.dropout_prob'],
            weight_normalize=hparams['transformer.weight_normalize'],
            use_bn=hparams['transformer.use_bn'],
        )

        self.conv3d = cnn3ds[hparams['conv3d.type']](
            in_channels=hparams['conv3d.in_channels'],
            desc_dim=hparams['conv3d.desc_dim'],
            fusion=hparams['conv3d.fusion'],
            dropout=hparams['conv3d.dropout'],
            dropout_prob=hparams['conv3d.dropout_prob'],
            weight_normalize=hparams['conv3d.weight_normalize'],
        )

    def forward(self, pcds, kpts, kpt_patches, lrfs, l2_normalize):
        # pcds: [(N, 3), ...]
        # kpts: [(K, 3), ...]
        # kpt_patches: [(K, P, 3), ...]
        # lrfs: [(K, 3, 3), ...]
        # l2_normalize: bool

        assert len(pcds) > 0 and len(kpts) == len(kpt_patches) <= len(pcds)

        all_descs = list()
        for i in range(len(kpts)):
            K = kpts[i].size(0)

            # Compute transformation for each keypoint
            pfeats = self.local_pointnet(kpt_patches[i])  # (K, LC)
            # (K, R, 4/6), (K, S)
            _, scales = self.transformer(pfeats)

            rotations = torch.unsqueeze(lrfs[i], 1)  # (K, 1, 3, 3)
            if rotations.size(1) != scales.size(1):
                # (K, S, 3, 3)
                rotations = rotations.expand(rotations.size(0), scales.size(1),
                                             rotations.size(2), rotations.size(3))

            # Voxelization
            # (K, S, V, 3)
            voxel_positions = transform_voxel_grids(self.voxel_grids, scales, rotations,
                                                    kpts[i])
            S, V, D = voxel_positions.shape[1:]
            voxel_positions = voxel_positions.view(-1, V, D)  # (K*S, V, 3)

            # (K*S, V)
            voxel_radii = self.voxel_radii.view(1, -1) * scales.view(-1, 1)

            # (K*S, V)
            voxel_vals = voxelize(pcds[i], voxel_positions, voxel_radii, self.sigma)
            voxel_vals = voxel_vals.view(K, S, 1, self.resolution, self.resolution,
                                         self.resolution)

            # 3D Conv
            kpt_descs = self.conv3d(voxel_vals, l2_normalize)  # (K, C')

            all_descs.append(kpt_descs)
        return all_descs


class UPDescUniScale(nn.Module):

    def __init__(self, **hparams):
        super().__init__()

        # Configs
        self.hparams = hparams

        # Desc extractor
        self.edge_length = hparams['voxelization.edge_length']
        self.resolution = hparams['voxelization.resolution']
        self.sigma = hparams['voxelization.sigma']

        # Constant variables
        voxel_size = self.edge_length / self.resolution

        voxel_grids = create_voxel_grids(self.edge_length, self.resolution)
        voxel_grids = np.reshape(voxel_grids, (-1, 3))
        self.register_buffer('voxel_grids', torch.from_numpy(voxel_grids))  # (V, 3)

        voxel_radius = voxel_size / 2.0
        voxel_radii = np.ones((len(voxel_grids),), dtype=np.float32) * voxel_radius
        self.register_buffer('voxel_radii', torch.from_numpy(voxel_radii))  # (V, )

        # Model sub-modules
        self.local_pointnet = None
        self.transformer = None
        self.voxel_scales = nn.Parameter(
            torch.as_tensor([0.5 * hparams['transformer.max_scale']], dtype=torch.float32))

        self.conv3d = cnn3ds[hparams['conv3d.type']](
            in_channels=hparams['conv3d.in_channels'],
            desc_dim=hparams['conv3d.desc_dim'],
            fusion=hparams['conv3d.fusion'],
            dropout=hparams['conv3d.dropout'],
            dropout_prob=hparams['conv3d.dropout_prob'],
            weight_normalize=hparams['conv3d.weight_normalize'],
        )

    def forward(self, pcds, kpts, kpt_patches, lrfs, l2_normalize):
        # pcds: [(N, 3), ...]
        # kpts: [(K, 3), ...]
        # kpt_patches: [(K, P, 3), ...]
        # lrfs: [(K, 3, 3), ...]
        # l2_normalize: bool

        assert len(pcds) > 0 and len(kpts) == len(kpt_patches) <= len(pcds)

        all_descs = list()
        for i in range(len(kpts)):
            K = kpts[i].size(0)

            scales = torch.clamp(self.voxel_scales,
                                 min=self.hparams['transformer.min_scale'],
                                 max=self.hparams['transformer.max_scale'])
            scales = scales.view(1, 1).expand(K, 1)

            rotations = torch.unsqueeze(lrfs[i], 1)  # (K, 1, 3, 3)
            if rotations.size(1) != scales.size(1):
                # (K, S, 3, 3)
                rotations = rotations.expand(rotations.size(0), scales.size(1),
                                             rotations.size(2), rotations.size(3))

            # Voxelization
            # (K, S, V, 3)
            voxel_positions = transform_voxel_grids(self.voxel_grids, scales, rotations,
                                                    kpts[i])
            S, V, D = voxel_positions.shape[1:]
            voxel_positions = voxel_positions.view(-1, V, D)  # (K*S, V, 3)

            # (K*S, V)
            voxel_radii = self.voxel_radii.view(1, -1) * scales.view(-1, 1)

            # (K*S, V)
            voxel_vals = voxelize(pcds[i], voxel_positions, voxel_radii, self.sigma)
            voxel_vals = voxel_vals.view(K, S, 1, self.resolution, self.resolution,
                                         self.resolution)

            # 3D Conv
            kpt_descs = self.conv3d(voxel_vals, l2_normalize)  # (K, C')

            all_descs.append(kpt_descs)
        return all_descs


class UPDescPCPNet(nn.Module):

    def __init__(self, **hparams):
        super().__init__()

        # Configs
        self.hparams = hparams

        # Desc extractor
        self.model = PCPNet(num_points=hparams['data.num_points_per_sample'],
                            output_dim=32,
                            use_point_stn=True,
                            use_feat_stn=True,
                            sym_op='max',
                            point_tuple=1)

    def forward(self, pcds, kpts, kpt_patches, lrfs, l2_normalize):
        # pcds: [(N, 3), ...]
        # kpts: [(K, 3), ...]
        # kpt_patches: [(K, P, 3), ...]
        # lrfs: [(K, 3, 3), ...]
        # l2_normalize: bool

        hparams = self.hparams
        assert len(kpts) == len(kpt_patches) == len(lrfs) > 0
        patch_radius = hparams['data.sample_radius']

        all_descs = list()
        for i in range(len(kpts)):
            # (K, P, 3), rotated by LRF
            xyz = (kpt_patches[i] - torch.unsqueeze(kpts[i], 1)) @ lrfs[i]
            xyz = xyz / patch_radius
            descs, _, _, _ = self.model(torch.transpose(xyz, 2, 1))
            assert descs.dim() == 2 and descs.size(0) == xyz.size(0)
            descs = descs / torch.norm(descs, p=2, dim=1, keepdim=True)
            all_descs.append(descs)
        return all_descs


MODELS = {d.__name__: d for d in [UPDesc, UPDescUniScale, UPDescPCPNet]}
