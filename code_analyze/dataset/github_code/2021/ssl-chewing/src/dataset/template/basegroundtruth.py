from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np


class BaseGroundTruth(ABC):
    """
    A base class for creating ground truth values.

    Child classes should implement the following methods: ``get_label``.
    """

    @abstractmethod
    def get_label(self, idx: int, t1: float, t2: float) -> Union[bool, int, List[int], np.ndarray]:
        """
        Return the ground truth for the interval [``t1``, ``t2``] of the ``idx``-th partition. Ground-truth values are
        free to be decided by the implementation.

        :param idx: The partition index
        :param t1: The start of the time interval
        :param t2: The stop of the time interval
        :return: The ground-truth value
        """
        raise NotImplementedError("Method not implemented")


class NoneGroundTruth(BaseGroundTruth):
    """
    Dummy implementation of ``BaseGroundTruth`` that always returns ``None``.
    """

    def get_label(self, idx: int, t1: float, t2: float):
        return None
