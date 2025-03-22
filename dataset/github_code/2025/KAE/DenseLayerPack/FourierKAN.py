import torch
import torch.nn as nn
import numpy as np


class NaiveFourierKANLayer(nn.Module):
    def __init__(
        self, inputdim, outdim, gridsize=5, addbias=True, smooth_initialization=False
    ):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        grid_norm_factor = (
            (torch.arange(gridsize) + 1) ** 2
            if smooth_initialization
            else np.sqrt(gridsize)
        )
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize)
            / (np.sqrt(inputdim) * grid_norm_factor)
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize),
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum(
            "dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs
        )
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y

    def regularization_loss():
        pass
