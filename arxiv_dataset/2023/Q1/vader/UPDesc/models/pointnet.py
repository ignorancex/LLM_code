import functools
import logging
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# PointNet/PointNet++ backbones
# Ref:
# - https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/models
# ---------------------------------------------------------------------------- #
class PointNetBase(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn, use_xyz, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.use_xyz = use_xyz
        # Additional arguments
        for k, w in kwargs.items():
            setattr(self, k, w)

        self._build()

    def _build(self):
        pass

    def _split_input(self, pcd):
        # pcd: (B, N, 3+)

        # (B, N, 3)
        xyz = pcd[..., 0:3].contiguous()
        # (B, C, N) or ()
        features = pcd[..., 3:].transpose(1, 2).contiguous() if pcd.size(-1) > 3 else None
        return xyz, features

    def forward(self, pcds):
        # pcds: (B, N, 3+)
        pass


class EncoderMSG(PointNetBase):
    """PointNet++ backbone for global feature extraction for a single point cloud
    """

    def _build(self):
        assert hasattr(self, 'sa_npoints')
        assert hasattr(self, 'sa_radii')
        assert hasattr(self, 'sa_nsamples')

        self.SA_modules = nn.ModuleList()

        lid = 0
        ch = self.in_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.sa_npoints[lid],
                radii=self.sa_radii[lid],
                nsamples=self.sa_nsamples[lid],
                mlps=[
                    [ch, 32, 32, 64],
                    [ch, 64, 64, 128],
                    [ch, 64, 96, 128],
                ],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))

        lid = 1
        ch = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.sa_npoints[lid],
                radii=self.sa_radii[lid],
                nsamples=self.sa_nsamples[lid],
                mlps=[
                    [ch, 64, 64, 128],
                    [ch, 128, 128, 256],
                    [ch, 128, 128, 256],
                ],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))

        lid = 2
        ch, och = 128 + 256 + 256, self.out_channels
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[ch, 256, 256, och],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))

    def forward(self, pcds):
        # pcds: (B, N, 3+)

        xyz, features = self._split_input(pcds)
        if features is None and not self.use_xyz:
            features = xyz.transpose(1, 2).contiguous()
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return torch.squeeze(features, dim=-1)  # (B, C)


class EncoderMSGLite(EncoderMSG):

    def _build(self):
        assert hasattr(self, 'sa_npoints')
        assert hasattr(self, 'sa_radii')
        assert hasattr(self, 'sa_nsamples')

        self.SA_modules = nn.ModuleList()

        ch = self.in_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.sa_npoints,
                radii=self.sa_radii,
                nsamples=self.sa_nsamples,
                mlps=[
                    [ch, 32, 32, 64],
                    [ch, 64, 64, 128],
                    [ch, 64, 96, 128],
                ],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))

        ch, och = 64 + 128 + 128, self.out_channels
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[ch, 256, 256, och],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))


class EncoderSSGLite(EncoderMSG):

    def _build(self):
        assert hasattr(self, 'sa_npoints')
        assert hasattr(self, 'sa_radii')
        assert hasattr(self, 'sa_nsamples')

        self.SA_modules = nn.ModuleList()

        ch = self.in_channels
        self.SA_modules.append(
            PointnetSAModule(
                npoint=self.sa_npoints,
                radius=self.sa_radii,
                nsample=self.sa_nsamples,
                mlp=[ch, 64, 96, 128],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))

        ch, och = 128, self.out_channels
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[ch, 256, 256, och],
                bn=self.use_bn,
                use_xyz=self.use_xyz,
            ))


def get_encoder(encoder_type, data):
    """
    Args:
        encoder_type (str): ['msg', 'msg_lite', 'ssg_lite']
        data (str): ['match3d', 'modelnet40'] 

    Returns:
        class:
    """
    if data == 'match3d':
        if encoder_type == 'msg':
            sa_npoints = [512, 128]
            sa_radii = [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8]]
            sa_nsamples = [[16, 32, 128], [32, 64, 128]]
            encoder_cls = EncoderMSG
        elif encoder_type == 'msg_lite':
            sa_npoints = 32
            sa_radii = [0.1, 0.2, 0.4]
            sa_nsamples = [64, 128, 256]
            encoder_cls = EncoderMSGLite
        elif encoder_type == 'ssg_lite':
            sa_npoints = 32
            sa_radii = 0.1
            sa_nsamples = 64
            encoder_cls = EncoderSSGLite
        else:
            raise RuntimeError('Encoder type - {} is not supported.'.format(encoder_type))
    elif data == 'modelnet40':
        if encoder_type == 'ssg_lite':
            sa_npoints = 16
            sa_radii = 0.1
            sa_nsamples = 32
            encoder_cls = EncoderSSGLite
        else:
            raise RuntimeError('Encoder type - {} is not supported.'.format(encoder_type))
    else:
        raise RuntimeError('Dataset - {} is not configured.'.format(data))

    return functools.partial(encoder_cls,
                             sa_npoints=sa_npoints,
                             sa_radii=sa_radii,
                             sa_nsamples=sa_nsamples)
