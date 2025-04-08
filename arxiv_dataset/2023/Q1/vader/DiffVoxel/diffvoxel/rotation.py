from __future__ import division
from __future__ import print_function

import os.path as osp
import torch
import torch.nn.functional as F

import diffvoxel._ext as cu


class QuatToMatFunc(torch.autograd.Function):
    """Ref:
    - https://github.com/nnaisense/deep-iterative-surface-normal-estimation/tree/master/cuda
    """

    @staticmethod
    def forward(ctx, x):
        """
        Args:
            ctx :
            x (torch.Tensor): (*, 4), unit, x0 + x1 * i + x2 * j + x3 * k

        Returns:
            torch.Tensor: (*, 3, 3)
        """
        assert x.size(-1) == 4

        y = torch.zeros(x.shape[:-1] + (3, 3), dtype=x.dtype, device=x.device)
        cu.quat_to_mat_fw(x.contiguous(), y.contiguous())

        # Save for backward
        ctx.save_for_backward(x)

        return y

    @staticmethod
    def backward(ctx, g_y):
        """
        Args:
            ctx :
            g_y (torch.Tensor): (*, 3, 3)

        Returns:
            torch.Tensor: (*, 4)
        """
        x, = ctx.saved_tensors
        g_x = None
        if ctx.needs_input_grad[0]:
            g_x = torch.zeros_like(x)
            cu.quat_to_mat_bw(x.contiguous(), g_y.contiguous(), g_x.contiguous())
        return g_x


quat_to_mat = QuatToMatFunc.apply


def ortho6d_to_mat(ortho6d):
    """Ref:
    - https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py

    Args:
        ortho6d (torch.Tensor): (*, 6)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    shape = list(ortho6d.size())  # (*, 6)

    x_raw = ortho6d[..., 0:3]  # (*, 3)
    y_raw = ortho6d[..., 3:6]  # (*, 3)

    x = F.normalize(x_raw, p=2, dim=-1)  # (*, 3)
    z = torch.cross(x, y_raw, dim=-1)  # (*, 3)
    z = F.normalize(z, p=2, dim=-1)  # (*, 3)
    y = torch.cross(z, x, dim=-1)  # (*, 3)

    matrix = torch.stack((x, y, z), dim=len(shape))
    return matrix
