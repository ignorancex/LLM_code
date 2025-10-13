from typing import Any, Literal, Tuple

import torch.nn as nn

Array = Any


class Architecture:
    """Specifies a neural network architecture.
    
    The architecture is not actually instantiated until create() is called.
    """

    def create(self, input_shape: Tuple[int], output_dim: int) -> nn.Module:
        """Actually instantiate the architecture.

        Args:
          input_shape (Tuple[int]): the shape of the network input
          output_dim (int): the dimension of the network output

        Returns:
          a PyTorch module for the network
        """
        raise NotImplementedError()


ActivationList = Literal["gelu", "silu"]

_activation_dict = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}

def make_activation(activation: str) -> nn.Module:
    return _activation_dict[activation]()
