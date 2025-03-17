import numpy as np
import torch
import torch.nn as nn


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1, device=rot.device)  # T,1
    ones = torch.ones(rot.shape[0], 1, device=rot.device)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


class random_rot(nn.Module):
    """
    data_numpy: C,T,V,M
    """

    def __init__(self, theta=0.3):
        super().__init__()
        self.theta = theta

    def forward(self, sample):
        if isinstance(sample, np.ndarray):
            data_torch = torch.from_numpy(sample)
        else:
            data_torch = sample
        device = data_torch.device
        C, T, V, M = data_torch.shape
        data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
        rot = torch.zeros(3, device=device).uniform_(-self.theta, self.theta)
        rot = torch.stack([rot, ] * T, dim=0)
        rot = _rot(rot)  # T,3,3
        data_torch = torch.matmul(rot, data_torch)  # rot.float() added
        data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

        return data_torch


class ntu_to_tensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tensor(x) if not isinstance(x, torch.Tensor) else x
