from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras.utils import to_categorical

from dataset.template.commons import PureAbstractError


class BaseLabelTransform(ABC):

    @abstractmethod
    def transform_batch(self, x: np.ndarray) -> np.ndarray:
        raise PureAbstractError()


class CategoricalLabelTransform(BaseLabelTransform):
    def __init__(self, noof_classes: int):
        assert isinstance(noof_classes, int)

        self._noof_classes: int = noof_classes

    def transform_batch(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        return to_categorical(x, self._noof_classes)


class NumpyCastLabelTransform(BaseLabelTransform):
    def __init__(self, dtype: type):
        assert isinstance(dtype, type)

        self._dtype: type = dtype

    def transform_batch(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        return x.astype(self._dtype)
