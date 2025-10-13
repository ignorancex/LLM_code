import torch
import torch.nn as nn
from math import sqrt

from ..configs import ActivationType
from ..layers import get_activation


class MLPLayer(nn.Module):
    """Implements a single MLP layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        activation (torch.nn.Module): Activation function. If None, defaults to
            ReLU activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        is_first=False,
        use_bias=True,
        activation: ActivationType = "relu",
        random_projection: bool = False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation, init, first_init = get_activation(activation)
        if first_init is not None and self.is_first:
            first_init(self.linear)
        else:
            init(self.linear)

        seed = 123
        if random_projection:
            # if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random_matrix = torch.randn_like(self.linear.weight.data)
            random_matrix /= sqrt(dim_in)

            self.linear.weight.data = random_matrix
            self.linear.weight.requires_grad = False
            # self.linear.weight.requires_grad_(False)
            if use_bias:
                random_matrix = torch.randn_like(self.linear.bias.data) / sqrt(dim_in)
                self.linear.bias.data = random_matrix
                self.linear.bias.requires_grad = False
                # self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        out = self.linear(x.to(self.linear.weight.dtype))
        out = self.activation(out)
        return out


class LoraLinear(nn.Linear):
    def __init__(self, in_features, out_features, lora_rank, omega):
        super().__init__(in_features, out_features)
        self.omega = omega
        self.g = in_features**0.5
        self.U = nn.Parameter(torch.zeros(out_features, lora_rank).cuda())
        nn.init.kaiming_uniform_(self.U, 5**0.5)
        self.V = nn.Parameter(torch.zeros(lora_rank, in_features).cuda())
        self.weight.requires_grad = False

    @classmethod
    def from_linear(cls, m: nn.Linear, lora_rank, omega):
        # For some reason nn.Linear.in_features is a tensor?
        new = cls(m.in_features.item(), m.out_features, lora_rank, omega)  # type: ignore
        new.load_state_dict(m.state_dict(), strict=False)
        return new

    def forward(self, input: torch.Tensor):
        adapter = torch.sin(self.omega * self.U @ self.V) / self.g
        new_weight = self.weight + adapter
        return nn.functional.linear(input, new_weight, self.bias)
