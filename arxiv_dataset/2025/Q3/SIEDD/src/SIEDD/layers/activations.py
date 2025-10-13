import torch
import torch.nn as nn
from ..utils.layer_utils import (
    init_weights_linear,
    init_weights_xavier,
    init_weights_selu,
    init_weights_elu,
    sine_init,
    first_layer_sine_init,
)
from typing import Callable
from ..configs import ActivationType


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(
        self,
        w0: float = 30.0,
        c=6.0,
        finer: bool = False,
        abcd: torch.Tensor | None = None,
        power: float | None = None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c  # std of the uniform distribution
        self.finer = finer
        self.power = power
        self.register_buffer("abcd", abcd)

    def forward(
        self,
        x: torch.Tensor,
        incode_params: torch.Tensor | None = None,
        decoder_idx=None,
    ):
        if decoder_idx is None:
            decoder_idx = slice(None)
        if incode_params is None and self.abcd is not None:
            incode_params = self.abcd
        if incode_params is not None:
            self.abcd = incode_params.detach()
            # a, b, c, d are of shape (N,)
            a = torch.exp(incode_params[:, 0])
            b = torch.exp(incode_params[:, 0])
            c = incode_params[:, 0]
            d = incode_params[:, 0]

            if x.ndim == 3:
                a = a[decoder_idx, None, None]
                b = b[decoder_idx, None, None]
                c = c[decoder_idx, None, None]
                d = d[decoder_idx, None, None]

            if self.finer:
                x = x * (torch.abs(x.detach()) + 1)
            return a * torch.sin(b * self.w0 * x + c) + d

        if self.finer:
            x = x * (torch.abs(x.detach()) + 1)
        return torch.sin(self.w0 * x)


class LinearSine(Sine):
    def forward(
        self,
        x: torch.Tensor,
        incode_params: torch.Tensor | None = None,
        decoder_idx=None,
    ):
        x = (self.w0 / torch.pi) * x - 0.5
        x = 2 * torch.abs((x % 2) - 1) - 1
        if self.power is not None:
            x = x**self.power
        return x


def get_activation(activation: ActivationType):
    """
    Returns the activation function and optionally its initialization functions based on the provided name.

    :param activation: Name of the activation function.
    :param return_init: If True, returns a tuple of (activation_function, init_function, first_layer_init_function).
    :param w0: Frequency for Sine activation (default: 30.0).
    :param c: Scaling factor for Sine activation (default: 6.0).
    :return: Activation function or tuple of (activation_function, init_function, first_layer_init_function).
    """
    # NOTE: Using Kaiming init for ReLU resulted in worse results, just using linear here
    activations: dict[str, tuple[nn.Module, Callable, Callable | None]] = {
        "none": (nn.Identity(), init_weights_linear, None),
        "linear": (nn.Identity(), init_weights_linear, None),
        "relu": (nn.ReLU(inplace=True), init_weights_linear, None),
        "leakyrelu": (nn.LeakyReLU(inplace=True), init_weights_linear, None),
        "tanh": (nn.Tanh(), init_weights_xavier, None),
        "sigmoid": (nn.Sigmoid(), init_weights_xavier, None),
        "selu": (nn.SELU(inplace=True), init_weights_selu, None),
        "elu": (nn.ELU(inplace=True), init_weights_elu, None),
        "softplus": (nn.Softplus(), init_weights_linear, None),
        "gelu": (nn.GELU(), init_weights_linear, None),
    }
    if isinstance(activation, str):
        act = activation.lower()
        if act not in activations:
            raise ValueError(f"Unknown activation function {act}")
        act_func, init_func, first_layer_init = activations[act]
    else:
        sine_type = LinearSine if activation.linear else Sine
        act_func = sine_type(
            activation.w0, activation.c, activation.finer, power=activation.power
        )
        init_func = sine_init(activation.w0)
        first_layer_init = first_layer_sine_init

    return act_func, init_func, first_layer_init
