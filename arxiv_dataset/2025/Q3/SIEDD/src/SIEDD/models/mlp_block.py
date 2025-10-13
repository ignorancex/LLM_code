import torch.nn as nn
import einops
import torch

from ..layers import MLPLayer, Sine
from .base_model import BaseModel
from ..configs import EncoderConfig, CoordXConfig, MLPConfig


class MLPBlock(BaseModel):
    """
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        num_layers (int): Number of layers.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        dim_out: int,
        data_shape: list[int],
        cfg: EncoderConfig,
        use_first_init: bool = True,
    ):
        super().__init__(cfg, data_shape)
        if isinstance(cfg.net, MLPConfig):
            self.net_cfg = cfg.net
        else:
            raise ValueError("MLPConfig Needed")
        # dim_in inside base_model
        self.dim_out = dim_out
        self.num_layers = self.net_cfg.num_layers
        self.dim_hidden = self.net_cfg.dim_hidden
        self.use_bias = self.net_cfg.use_bias

        self.final_activation = self.net_cfg.final_activation
        self.cfg = cfg

        self.random_projection = self.net_cfg.random_projection
        self.patch_size = self.cfg.patch_size

        first_dim_in = self.cfg.net.pos_encode_cfg.dim_out
        if isinstance(self.cfg.net.pos_encode_cfg, CoordXConfig):
            use_first_init = False
        layers = []
        if self.num_layers == -1 and dim_out != self.dim_in:
            raise ValueError("Different in and out dims for identity MLP")
        if self.net_cfg.bottleneck:
            # TODO: dims should go first_dim_in -> dim_hidden -> dim_out
            dims = torch.linspace(
                first_dim_in, self.dim_hidden, self.num_layers + 2, dtype=torch.int
            )
        else:
            dims = torch.full((self.num_layers + 2,), self.dim_hidden)
            if self.num_layers != -1:
                dims[0] = first_dim_in
                dims[-1] = dim_out
        for ind, (in_feats, out_feats) in enumerate(zip(dims[:-1], dims[1:])):
            is_first = ind == 0
            act = (
                self.net_cfg.final_activation
                if ind == self.num_layers - 1
                else self.net_cfg.activation
            )
            layers.append(
                MLPLayer(
                    dim_in=in_feats,
                    dim_out=out_feats,
                    use_bias=self.use_bias,
                    is_first=is_first and use_first_init,
                    activation=act,
                    random_projection=self.random_projection,
                )
            )
        self.net = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, preprocess_output=True, abcd: torch.Tensor | None = None
    ):
        out: torch.Tensor = self.positional_encoder(x)
        for layer in self.net:
            if isinstance(layer, Sine):
                out = layer(out, abcd)
            else:
                out = layer(out)
        if preprocess_output:
            out = self.process_output(out)
        return out

    def process_output(self, out: torch.Tensor):
        if self.patch_size is not None:
            if out.ndim == 2:
                out = einops.rearrange(
                    out,
                    "n (c pw ph gs) -> gs n c pw ph",
                    c=3,
                    pw=self.patch_size,
                    ph=self.patch_size,
                )
            else:
                out = out.reshape(
                    out.size(0), out.size(1), 3, self.patch_size, self.patch_size
                )
        else:
            if out.ndim == 2:
                out = einops.rearrange(out, "n (c gs) -> gs n c", c=3)
            else:
                out = out.reshape(-1, out.size(1), 3)
        return out
