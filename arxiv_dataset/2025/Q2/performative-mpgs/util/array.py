from jax import Array
from jax import numpy as np


def rescale(arr: Array) -> Array:
    """Rescales array values to [0, 1]"""
    min_val = np.min(arr)
    max_val = np.max(arr)

    return (arr - min_val) / (max_val - min_val)
