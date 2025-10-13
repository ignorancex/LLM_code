import torch
from torch import nn
import torch.nn.functional as f
from .utils import PI, SQRT2, deg2rad
from .filters import RampFilter


class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float, device=torch.device('cuda')):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        self.device = device

        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = f.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape

        L, _, _, _ = self.all_grids.shape

        grid = self.all_grids.to(x.device).view(L * W, W, 2).expand(N, -1, -1, -1)
        x_sampled = f.grid_sample(x, grid, align_corners=True)
        out = x_sampled.view(N, C, L, W, W).sum(dim=3).transpose(-1, -2)
        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())

        rad = deg2rad(angles)
        c, s = rad.cos(), rad.sin()
        R = torch.stack((torch.stack((c, s, torch.zeros_like(c)), dim=-1),
                         torch.stack((-s, c, torch.zeros_like(c)), dim=-1)), dim=-2)
        return f.affine_grid(R, (R.shape[0], 1, grid_size, grid_size), align_corners=True)


class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float, device=torch.device('cuda')):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        self.device = device

        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
        if None in [self.ygrid, self.xgrid, self.all_grids]:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        x = self.filter(x).to(self.device)

        N, C, W, _ = x.shape
        L, _, _, _ = self.all_grids.shape
        grid = self.all_grids.to(x.device).view(L * W, W, 2).expand(N, -1, -1, -1)
        x_sampled = f.grid_sample(x, grid, align_corners=True)
        reco = x_sampled.view(N, C, L, W, W).sum(dim=2)

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = f.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1])).to(self.device)

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = f.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype).to(self.device)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin().to(self.device)
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())

        X = torch.linspace(-1.0, 1.0, len(angles)).unsqueeze(-1).unsqueeze(-1).expand(-1, grid_size, grid_size)
        rad = deg2rad(angles).unsqueeze(-1).unsqueeze(-1)
        c, s = rad.cos(), rad.sin()
        Y = self.xgrid.unsqueeze(0) * c - self.ygrid * s
        return torch.stack((X.to(self.device), Y.to(self.device)), dim=-1)
