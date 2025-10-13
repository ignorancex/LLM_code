import torch.nn as nn
import torch

from . import get_activation
from ..configs import SirenNeRVConfig, EncoderConfig


class NervBlock(nn.Module):
    def __init__(self, dim_in, dim_out, cfg: EncoderConfig, sncfg: SirenNeRVConfig):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.cfg = cfg
        self.sncfg = sncfg

        self.up_sample = sncfg.up_sample

        self.activation = get_activation(sncfg.nerv_act)  # ?

        self.nerv_block = nn.Sequential(
            nn.Conv2d(
                self.dim_in,
                self.dim_out,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.PixelShuffle(self.up_sample),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.nerv_block(x)
        return out
