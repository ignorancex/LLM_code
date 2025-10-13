from abc import ABC, abstractmethod
from typing import Iterable, Union

import torch


class Propagator(ABC):

    layers: Iterable[str]

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_layers(self, layers: Union[str, Iterable[str]]) -> None:
        pass

    @abstractmethod
    def get_predictions(self, img_tensor: torch.Tensor) -> None:
        pass

    @abstractmethod
    def get_activations(self) -> None:
        pass

    @abstractmethod
    def get_gradients(self) -> None:
        pass


class PropagatorTransformer(Propagator):

    pass