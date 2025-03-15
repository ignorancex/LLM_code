from abc import ABC, abstractmethod

import numpy as np

from dataset.template.commons import PureAbstractError


class BaseNormalizer(ABC):
    """
    Base class for normalizers.
    """

    @abstractmethod
    def normalize(self, x: np.ndarray) -> np.ndarray:
        raise PureAbstractError()


class ZeroMeanNormalizer(BaseNormalizer):
    def normalize(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        return x - np.mean(x)


class StandardizerNormalizer(BaseNormalizer):
    def __init__(self, min_sigma: float = 1e-9):
        assert isinstance(min_sigma, float)

        # self._min_sigma: np.ndarray = np.array(min_sigma)
        self._min_sigma: float = min_sigma

    def normalize(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        sigma = np.max((np.std(x), self._min_sigma))

        return (x - np.mean(x)) / sigma
