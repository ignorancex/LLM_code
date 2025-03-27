"""
Module that provides various reshapers.

Each reshaper class should provide a function called 'reshape' with the following signature:
    reshape(x)
where x is a numpy ndarray (len(x.shape) should be equal to 1), and should return an ndarray
that contains the reshaped version of x.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dataset.template.commons import PureAbstractError
from utilities.numpyutils import is_numpy_2d_vector
from utilities.typingutils import is_typed_list


class BaseReshaper(ABC):
    """
    Base class for reshapers.
    """

    @abstractmethod
    def get_output_shape(self) -> List[int]:
        raise PureAbstractError()

    @abstractmethod
    def reshape(self, x: np.ndarray) -> np.ndarray:
        raise PureAbstractError()


class DefaultReshaper(BaseReshaper):
    """
    A reshaper that ensures that 1D arrays are in fact 2D with the second dimension being equal to 1.
    """

    def __init__(self, input_length: List[int] = None):
        assert is_typed_list(input_length, int) or input_length is None

        self._input_length: List[int] = input_length

    def get_output_shape(self) -> List[int]:
        return self._input_length + [1]

    def reshape(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        elif len(x.shape) == 2:
            assert x.shape[1] == 1
        else:
            raise ValueError("Incompatible shape for x")

        return x


class BufferReshaper(BaseReshaper):
    """
    A reshaper that reshapes a 1D window to an array of sub-windows (similar to MATLAB's buffer function).
    """

    def __init__(self, input_length: int, wsize: int, wstep: int):
        """
        Create a reshaper that buffers the input window.

        :param input_length: The size (length) of the input window.
        :param wsize: The size (length) of the sub-windows.
        :param wstep: The step for the sub-windows.
        """
        assert isinstance(input_length, int)
        assert isinstance(wsize, int)
        assert isinstance(wstep, int)

        self._input_length = input_length
        self._wsize = wsize
        self._wstep = wstep

        self._start_idxs = np.arange(0, input_length - wsize + 1, wstep, dtype=int)
        self._noof_windows = len(self._start_idxs)

    def get_output_shape(self) -> List[int]:
        return [self._noof_windows, self._wsize]

    def reshape(self, x: np.ndarray) -> np.ndarray:
        assert is_numpy_2d_vector(x)

        y = np.empty((self._noof_windows, self._wsize, 1))
        for i, idx in enumerate(self._start_idxs):
            y[i, :] = x[idx:idx + self._wsize]

        return y
