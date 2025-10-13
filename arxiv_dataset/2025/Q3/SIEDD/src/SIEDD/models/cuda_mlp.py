from ..utils import requires_tcnn, tcnn
from ..configs import EncoderConfig, CudaMLPConfig, SineConfig
from . import BaseModel

import torch

"""
An MLP class using tinycudann library - Pure cuda implementation for speedup.
"""


@requires_tcnn()
class CudaMLP(BaseModel):
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
    ):
        super().__init__(cfg, data_shape)
        super().__init__(cfg, data_shape)
        net_cfg = cfg.net
        if not isinstance(net_cfg, CudaMLPConfig):
            raise ValueError("CudaMLPConfig Needed")
        self.net_cfg = net_cfg
        self.patch_size = self.cfg.patch_size
        dim_hidden = self.net_cfg.dim_hidden
        if isinstance(net_cfg.activation, SineConfig) or isinstance(
            net_cfg.final_activation, SineConfig
        ):
            raise ValueError("Do not use tinycudann with sine activation")
        if dim_hidden in [16, 32, 64, 128]:
            otype = "FullyFusedMLP"
        else:
            otype = "CutlassMLP"

        to_tcnn_activation = {
            "none": "None",
            "linear": "None",
            "relu": "ReLU",
            "leakyrelu": "LeakyReLU",
            "tanh": "Tanh",
            "sigmoid": "Sigmoid",
            "softplus": "Softplus",
            "sine": "Sine",
        }
        tcnn_config = {
            "otype": otype,
            "activation": to_tcnn_activation[net_cfg.activation],
            "output_activation": to_tcnn_activation[net_cfg.final_activation],
            "n_neurons": net_cfg.dim_hidden,
            "n_hidden_layers": net_cfg.num_layers,
        }
        self.net = tcnn.Network(self.dim_in, dim_out, tcnn_config)

    def forward(self, x: torch.Tensor, preprocess_output=True):
        pos_enc: torch.Tensor = self.positional_encoder(x)
        out = self.net(pos_enc)
        if preprocess_output:
            out = self.process_output(out)
        return out

    def process_output(self, out: torch.Tensor):
        if self.patch_size is not None:
            out = out.reshape(
                out.size(0), out.size(1), 3, self.patch_size, self.patch_size
            )
        else:
            out = out.reshape(-1, 1, 3)
        return out
