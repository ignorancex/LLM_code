import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_layer import BaseL0Layer

LayerMasks = tuple[Tensor, Tensor | None]
MaskedForwardOutput = tuple[Tensor, Tensor | None]


class UnstructuredL0Layer(BaseL0Layer):
    def __init__(self, l2_detach_gates: bool, droprate_init: float, temperature: float, *args, **kwargs):
        super().__init__(l2_detach_gates, droprate_init, temperature, *args, **kwargs)

        self.gates_shape = self.weight.shape
        self.group_size = 1
        self.weight_log_alpha = nn.Parameter(torch.empty_like(self.weight))

        # NOTE: bias sparsity not supported
        self.bias_log_alpha = None
        # self.bias_log_alpha = nn.Parameter(torch.empty_like(self.bias)) if self.bias is not None else None

        self.init_gates_parameters()

    def get_io_mask(self, weight_z) -> Tensor | None:
        if weight_z is None:
            # There is no mask
            return None

        # Only if all the entries associated with an output channel are zero, is the
        # output channel is masked.
        io_mask = torch.sum(weight_z, dim=1)

        # Ensure that io_mask is boolean
        io_mask = io_mask > 0 if io_mask is not None else None

        return io_mask

    def expected_sq_l2_norm(self, active_probs: torch.Tensor) -> torch.Tensor:
        if self.l2_detach_gates:
            active_probs = active_probs.detach()

        # We have one gate per weight, so the expected L2 norm is the sum of
        # the squared parameters weighted by the active probabilities.
        weight_exp_sq_norm = torch.sum(active_probs * self.weight.pow(2))

        bias_sq_norm = 0.0
        if self.bias is not None:
            bias_sq_norm = torch.sum(self.bias.pow(2))
            # NOTE: we detach the bias squared norm to avoid regularization of the bias
            bias_sq_norm = bias_sq_norm.clone().detach()

        return weight_exp_sq_norm + bias_sq_norm


class UnstructuredL0Linear(UnstructuredL0Layer, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> MaskedForwardOutput:
        is_test_time = not self.training
        weight_z, bias_z = self.sample_gates(is_test_time=is_test_time)
        weight, bias = self.get_parameters(weight_z, bias_z)
        out = F.linear(x, weight, bias)
        mask = self.get_io_mask(weight_z)
        return out, mask
