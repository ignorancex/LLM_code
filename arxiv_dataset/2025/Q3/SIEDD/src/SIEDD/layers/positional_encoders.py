import numpy as np

import torch
import torch.nn as nn

from .activations import get_activation
from ..configs import (
    FourierConfig,
    NeRFConfig,
    CoordXConfig,
    GaussianConfig,
    EncoderConfig,
    NoPosEncode,
    CudaHashgridConfig,
)
from ..utils.tinycudann import tcnn


class PosEncodingFourier(nn.Module):
    def __init__(self, cfg: FourierConfig):
        super().__init__()
        self.dim_in = cfg.dim_in
        self.dim_out = cfg.dim_out // 2
        self.scale = cfg.pos_scale

        torch.manual_seed(123)
        np.random.seed(123)

        param = torch.randn((self.dim_out, self.dim_in)) * self.scale
        if cfg.trainable:
            self.B = nn.Parameter(param)
        else:
            self.register_buffer("B", param, persistent=False)
        self.B: torch.Tensor
        self.param_shape = self.B.shape

        self.cached_projection: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, cache=False) -> torch.Tensor:
        if self.cached_projection is not None:
            return self.cached_projection
        x_proj = (2.0 * np.pi * x) @ self.B.t()
        random_fourier_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if cache:
            self.cached_projection = random_fourier_proj
        return random_fourier_proj


class PosEncodingNeRF(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(
        self,
        cfg: NeRFConfig,
    ):
        super().__init__()

        self.dim_in = cfg.dim_in
        self.include_coord = cfg.include_coord

        param = 2 ** torch.arange(cfg.param_size)[None, :, None] * np.pi
        if cfg.trainable:
            self.freq_bands = nn.Parameter(param)
        else:
            self.register_buffer("freq_bands", param, persistent=False)
        self.freq_bands: torch.Tensor

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # removes for loop over sine and cosine.
        # (n, 1, in) * (1, param_size, 1) -> (n, param_size, in)
        coords_pos_enc = coords.unsqueeze(1) * self.freq_bands
        # (n, param_size, in) -> (n, param_size * in)
        coords_pos_enc = coords_pos_enc.flatten(1, 2)
        sin = torch.sin(coords_pos_enc)
        cos = torch.cos(coords_pos_enc)

        # (n, param_size * in * 2)
        coords_pos_enc = torch.cat([sin, cos], -1)

        if self.include_coord:
            # (n, param_size * in * 2 + in)
            coords_pos_enc = torch.cat([coords, coords_pos_enc], -1)

        return coords_pos_enc


class PosEncodingCoordX(nn.Module):
    def __init__(
        self,
        dims: list[int],
        cfg: CoordXConfig,
    ):
        from SIEDD.models.generic_mlp import get_mlp

        super().__init__()
        cfg = cfg.model_copy(deep=True)
        self.dims = dims
        self.K = len(self.dims)
        self.dim_out = cfg.dim_out
        torch.manual_seed(123)
        np.random.seed(123)

        # Initial positional encoding
        self.pos_encoder: nn.Module
        if isinstance(cfg.net_cfg.pos_encode_cfg, NeRFConfig):
            self.pos_encoder = PosEncodingNeRF(cfg.net_cfg.pos_encode_cfg)
        elif isinstance(cfg.net_cfg.pos_encode_cfg, FourierConfig):
            self.pos_encoder = PosEncodingFourier(cfg.net_cfg.pos_encode_cfg)
        elif isinstance(cfg.net_cfg.pos_encode_cfg, NoPosEncode):
            self.pos_encoder = nn.Identity()
        else:
            raise ValueError("Invalid positional encoder for CoordX")

        # Parallel branches
        self.B1 = nn.Linear(cfg.net_cfg.pos_encode_cfg.dim_out, cfg.dim_out)
        self.B2 = nn.Linear(cfg.net_cfg.pos_encode_cfg.dim_out, cfg.dim_out)

        self.nonlin, init, first_init = get_activation(cfg.net_cfg.activation)
        if first_init is not None:
            first_init(self.B1)
            first_init(self.B2)
        else:
            init(self.B1)
            init(self.B2)
        cfg.net_cfg.pos_encode_cfg = NoPosEncode(
            dim_in=cfg.dim_out, dim_out=cfg.dim_out
        )
        self.hidden = get_mlp(
            dim_out=cfg.dim_out,
            data_shape=self.dims,
            cfg=EncoderConfig(net=cfg.net_cfg),
            use_first_init=False,
        )
        self.fusion_op = cfg.fusion_op

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        With no pos encoding:
        input = [(H, 1), (W, 1)]
        then we do (H, 1) @ (1, pos_out_dim), (W, 1) @ (1, pos_out_dim) (parallel branches)
        to get (H, pos_out_dim), (W, pos_out_dim).
        We then run each through linears to get similar shape
        Then we unsqueeze (H, 1, dim_hidden) and (1, W, dim_hidden)
        and fuse.

        With NeRF:
        input = [(H, 1), (W, 1)]
        We then do (H, 1) -> (H, pos_out_dim), (W, 1) -> (W, pos_out_dim) via NeRF
        then we do (H, pos_out_dim) @ (pos_out_dim, pos_out_dim) and
        (W, pos_out_dim) @ (pos_out_dim, pos_out_dim) (parallel branches)
        to get (H, pos_out_dim), (W, pos_out_dim).
        We then run each through linears to get same shape
        Then we unsqueeze (H, 1, dim_hidden) and (1, W, dim_hidden)
        and fuse.
        """

        if len(inputs) != 2:
            raise ValueError(f"Expected a list of 2 tensors, got {type(inputs)}")
        h: torch.Tensor
        w: torch.Tensor
        h, w = inputs
        # First do positional encoding if applicable
        h = self.pos_encoder(h)
        w = self.pos_encoder(w)

        # Now do parallel branches for each coord
        h, w = self.nonlin(self.B1(h)), self.nonlin(self.B2(w))
        # h, w = self.B1(h), self.B2(w)

        # Layers before fusion with shared parameters
        h = self.hidden(h, preprocess_output=False).unsqueeze(1)
        w = self.hidden(w, preprocess_output=False).unsqueeze(0)

        # Fusion
        if self.fusion_op == "+":
            z = h + w
        elif self.fusion_op == "*":
            z = h * w
        elif self.fusion_op == "mean":
            z = (h + w) / 2.0
        else:
            raise ValueError()

        # (H, W, dim_hidden) -> (HW, dim_hidden)
        z = z.reshape((-1, z.size(-1)))
        return z


class PosEncodingGaussian(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, dim_out*2, width, height].
    """

    def __init__(self, cfg: GaussianConfig):
        super().__init__()

        self._num_input_channels = cfg.dim_in
        self._dim_out = cfg.dim_out
        self._B = torch.randn((cfg.dim_in, cfg.dim_out // 2)) * cfg.pos_scale
        self.param_shape = self._B.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Expected 4D input (got {}D input)".format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels, (
            "Expected input to have {} channels (got {} channels)".format(
                self._num_input_channels, channels
            )
        )

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._dim_out)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class PosEncodingHashgrid(nn.Module if tcnn is None else tcnn.Encoding):  # type: ignore
    def __init__(self, dim_in: int, cfg: CudaHashgridConfig):
        # A wrapper around tcnn.Encoding
        dtype = torch.float16 if cfg.pos_dtype == "float16" else torch.float32
        super().__init__(dim_in, cfg.tcnn_config(), seed=123, dtype=dtype)
        gauss_init = cfg.hash_grid_gauss_init
        if gauss_init:
            mean = cfg.hash_grid_init_mean
            std = cfg.hash_grid_init_std
            print("Gaussian Initialization of hash params")
            with torch.no_grad():
                torch.nn.init.normal_(self.params, mean=mean, std=std)

    @torch.compiler.disable()
    def forward(self, x):  # type: ignore
        return super().forward(x)
