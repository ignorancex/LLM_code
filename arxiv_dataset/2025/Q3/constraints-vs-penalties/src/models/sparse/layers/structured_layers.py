import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_layer import BaseL0Layer

LayerMasks = tuple[Tensor, Tensor | None]
MaskedForwardOutput = tuple[Tensor, Tensor | None]


class StructuredL0Layer(BaseL0Layer):
    def __init__(self, l2_detach_gates: bool, droprate_init: float, temperature: float, *args, **kwargs):
        super().__init__(l2_detach_gates, droprate_init, temperature, *args, **kwargs)

        if isinstance(self, nn.Linear):
            self.gates_shape = self.in_features
            self.group_size = self.out_features

        self.weight_log_alpha = nn.Parameter(torch.empty(self.gates_shape, device=self.weight.device))

        # NOTE: bias sparsity not supported
        self.bias_log_alpha = None

        self.init_gates_parameters()

    def get_io_mask(self, weight_z) -> Tensor | None:
        if weight_z is None:
            # There is no mask
            return None

        if isinstance(self, nn.Linear):
            # Linear layers do input sparsity, so there is no output sparsity mask
            return None

        # For convolutional layers, ensure that io_mask is boolean
        io_mask = weight_z > 0

        return io_mask

    def expected_sq_l2_norm(self, active_probs: torch.Tensor) -> torch.Tensor:
        if self.l2_detach_gates:
            active_probs = active_probs.detach()

        # Get the squared L2 norm of each group of weights
        if isinstance(self, nn.Linear):
            w_group_sq_norm = torch.sum(self.weight.pow(2), dim=0)
        elif isinstance(self, nn.Conv2d):
            w_group_sq_norm = torch.sum(self.weight.pow(2), dim=(1, 2, 3))

        # Each group has a different proability of being active, so we weight the
        # squared L2 norm of each group by its active probability.
        weight_exp_sq_norm = torch.sum(active_probs * w_group_sq_norm)

        bias_sq_norm = 0.0
        if self.bias is not None:
            bias_sq_norm = torch.sum(self.bias.pow(2))
            # NOTE: we detach the bias squared norm to avoid regularization of the bias
            bias_sq_norm = bias_sq_norm.clone().detach()

        return weight_exp_sq_norm + bias_sq_norm


class StructuredL0Linear(StructuredL0Layer, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> MaskedForwardOutput:
        is_test_time = not self.training
        weight_z, bias_z = self.sample_gates(is_test_time=is_test_time)
        weight, bias = self.get_parameters(weight_z, bias_z)
        out = F.linear(x, weight, bias)
        mask = self.get_io_mask(weight_z)
        return out, mask
