import torch
import torch.nn as nn

from .base_model import BaseModel
from .mlp_block import MLPBlock
from ..layers.nerv_block import NervBlock
from ..configs import EncoderConfig, SirenNeRVConfig


class PosEncoding(nn.Module):
    def __init__(self, dim: int, num_frames: int, freq: float):
        super().__init__()
        assert dim > 1
        inv_freq = torch.zeros(dim)
        inv_freq[0::2] = torch.pow(1 / freq, torch.arange(dim - dim // 2))
        inv_freq[1::2] = torch.pow(1 / freq, torch.arange(dim // 2))
        pos_vec = inv_freq.unsqueeze(1) * torch.arange(num_frames).unsqueeze(0)
        pos_vec[1::2, :] += torch.pi / 2
        self.pos_encoding = nn.Parameter(
            torch.sin(pos_vec).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            requires_grad=False,
        )
        self.num_frames = num_frames
        assert self.pos_encoding.size() == (1, dim, num_frames, 1, 1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4
        N, C, H, W = x.size()
        out = x.unsqueeze(2) + self.pos_encoding
        return out.reshape(N, C * self.num_frames, H, W)


# TODO: Implement and Experiment?
class SirenNerv(BaseModel):
    def __init__(
        self,
        data_shape: list[int],
        cfg: EncoderConfig,
        sncfg: SirenNeRVConfig,
    ):
        super().__init__(cfg, data_shape)

        self.num_layers = sncfg.num_layers
        self.dim_hidden = sncfg.dim_hidden
        self.use_bias = sncfg.use_bias
        self.final_activation = sncfg.final_activation
        self.up_sample = sncfg.up_sample

        self.nerv_act = sncfg.nerv_act
        self.activation = sncfg.activation

        if cfg.patch_size is None:
            raise ValueError("SirenNerv: Patch size cannot be None")
        self.patch_size = cfg.patch_size

        self.expand_w = self.expand_h = self.patch_size // 2
        self.expand_ch = sncfg.expand_ch
        self.expand_dims = self.expand_w * self.expand_h * self.expand_ch

        self.mlp = MLPBlock(dim_out=self.expand_dims, data_shape=data_shape, cfg=cfg)

        self.nerv_in = self.expand_ch
        self.nerv_out = self.patch_size
        self.last_in = self.patch_size // (self.up_sample**2)

        self.pre_nerv = PosEncoding(self.expand_ch, 1, sncfg.num_freq)
        self.nerv_block = NervBlock(
            dim_in=self.nerv_in,
            dim_out=self.nerv_out,
            cfg=cfg,
            sncfg=sncfg,
        )

        self.last_layer = nn.Conv2d(
            self.last_in,
            3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=1,
        )

    def forward(self, x: torch.Tensor):
        enc: torch.Tensor = self.positional_encoder(x)
        out1: torch.Tensor = self.mlp(enc)
        out2: torch.Tensor = out1.view(
            out1.size(0), self.expand_ch, self.expand_h, self.expand_w
        )
        out3: torch.Tensor = self.pre_nerv(out2)
        out4: torch.Tensor = self.nerv_block(out3)
        out5: torch.Tensor = self.last_layer(out4)
        out6: torch.Tensor = out5.view(
            out5.size(0), 3, self.patch_size, self.patch_size
        )
        return out6
