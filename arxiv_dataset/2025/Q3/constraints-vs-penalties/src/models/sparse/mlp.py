from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from .layers import StructuredL0Linear, UnstructuredL0Linear


class L0MLP(nn.Module):
    """MLP with L0 regularization. If no hidden layers are specified, constructs
    a logistic regression model. It is built exclusively with L0Linear layers and
    activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sparsity_type: Literal["structure", "unstructured"] = "structured",
        layer_dims: tuple[int, ...] = (300, 100),
        l2_detach_gates: bool = False,
        temperature: float = 2.0 / 3.0,
        droprate_init: float = 0.5,
        use_bias: bool = True,
        act_fn_module: Callable = nn.ReLU,
    ):
        super().__init__()

        self.input_shape = (1, input_dim)
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims is not None else []
        self.num_classes = num_classes

        self.sparsity_type = sparsity_type
        self.use_bias = use_bias
        self.act_fn_module = act_fn_module

        linear_layer = StructuredL0Linear if sparsity_type == "structured" else UnstructuredL0Linear

        # --------------------- Construct Linear Layers ---------------------

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            # Use different (low) sparsity for input layer
            droprate_init_ = 0.2 if i == 0 else droprate_init

            layer = linear_layer(
                in_features=inp_dim,
                out_features=dimh,
                bias=use_bias,
                droprate_init=droprate_init_,
                l2_detach_gates=l2_detach_gates,
                temperature=temperature,
            )
            layers += [layer, self.act_fn_module()]

        # Input dim for output layer is different for Logistic regression.
        last_hidden = layer_dims[-1] if len(layer_dims) > 0 else input_dim
        layers.append(
            linear_layer(
                in_features=last_hidden,
                out_features=num_classes,
                bias=use_bias,
                droprate_init=droprate_init,
                l2_detach_gates=l2_detach_gates,
                temperature=temperature,
            )
        )

        self.fcs = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(-1, self.input_dim)

        for layer in self.fcs:
            if isinstance(layer, (StructuredL0Linear, UnstructuredL0Linear)):
                input = layer(input)[0]
            else:
                input = layer(input)
        return input
