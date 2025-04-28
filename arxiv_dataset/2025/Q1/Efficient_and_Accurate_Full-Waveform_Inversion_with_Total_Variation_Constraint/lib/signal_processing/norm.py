import numpy as np
from numpy import typing as npt


def L12_norm(signal: npt.NDArray):
    return np.sum(np.sqrt(np.sum(signal**2, axis=2)))
