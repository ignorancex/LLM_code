import abc
from typing import Tuple, Any
import torch
from torch import nn


class AbstractBaseImageEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def layer_shapes(self): # define the output dimension of the last layer
        raise NotImplementedError

    @abc.abstractmethod
    def code(self):
        raise NotImplementedError