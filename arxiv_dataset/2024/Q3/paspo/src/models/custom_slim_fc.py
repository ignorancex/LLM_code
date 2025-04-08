#this does not init the bias weights to 0.0 by default

from typing import Any
from torch import TensorType
import torch.nn as nn
from ray.rllib.models.utils import get_activation_fn

class CustomSlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Any = None,
        activation_fn: Any = None,
        use_bias: bool = True,
        bias_init: float = None,
    ):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size: Output size for FC Layer
            initializer: Initializer function for FC layer weights
            activation_fn: Activation function at the end of layer
            use_bias: Whether to add bias weights or not
            bias_init: Initalize bias weights to bias_init const
        """
        super(CustomSlimFC, self).__init__()
        layers = []
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None and False:
            initializer = nn.init.xavier_uniform_
        if False:
            initializer(linear.weight)
        if use_bias is True and bias_init is not None:
            nn.init.constant_(linear.bias, bias_init)
            
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)
