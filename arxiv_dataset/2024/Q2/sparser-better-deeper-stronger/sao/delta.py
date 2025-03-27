import torch
import torch.nn as nn
from itertools import product
from sao.ramanujan_constructions import Ramanujan_Constructions
from sao.base import Base


class Delta_Module(Ramanujan_Constructions, Base):
    def __init__(
        self,
        module: nn.Module,
        gain = 1.0,
        sigma_b = 0.0,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation: str = "relu",
        same_mask: bool = False,
        in_channels_0: int = 3,
        num_classes: int = 100,
    ):
        self.module = module
        self.sparsity = sparsity
        self.degree = degree
        self.num_classes = num_classes
        self.method = method
        self.activation = activation
        self.same_mask = same_mask
        self.gain = gain
        self.sigma_b = sigma_b
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            self.in_channels = module.in_channels
            self.out_channels = module.out_channels
        self.in_channels_0 = in_channels_0 # This is the number of input channels of the first convolutional layer

    def _sao_linear(self):
        constructor = self._ramanujan_structure()
        return constructor()

    def _sao_delta(self):
        constructor = self._ramanujan_structure()
        sao_matrix, sao_mask = constructor()
        sao_delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        sao_delta_weights[:, :, 1, 1] = sao_matrix
        sao_delta_mask = torch.zeros_like(self.module.weight).to("cuda")

        for i, j in product(
            range(self.module.out_channels), range(self.module.in_channels)
        ):
            sao_delta_mask[i, j] = sao_mask[i, j]

        return sao_delta_weights, sao_delta_mask

    def _delta(self) -> torch.tensor:
        weights = self._ortho_generator()
        delta_weights = torch.zeros_like(self.module.weight).to("cuda")
        delta_weights[:, :, 1, 1] = weights

        return delta_weights.to('cuda')

    def __call__(self):
        return (
            self._sao_delta()
            if (self.degree or self.sparsity)
            and self.in_channels != self.in_channels_0
            and self.method == "SAO"
            else self._delta()
        )


def Delta_Constructor(module, **kwargs):
    return Delta_Module(module, **kwargs)()

def Linear_Constructor(module, **kwargs):
    return Delta_Module(module, **kwargs)._sao_linear()
