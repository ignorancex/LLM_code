"""
Utility functions
"""

from numpy.typing import ArrayLike
import numpy as np

from sklearn.model_selection import train_test_split


def convert_to_onehot(
        xs: ArrayLike,
        n_classes: int = 0,
) -> np.ndarray:
    """Convert an integer-valued ordinal array to one-hot representation.

    If `n_classes == 0`, then
        it is inferred from the maximum integer value in `xs`.
    """
    xs = np.array(xs)
    n_classes = n_classes if n_classes >= 1 else max(xs) + 1
    xs_onehot = np.zeros((len(xs), n_classes))
    xs_onehot[np.arange(len(xs)), xs] = 1
    return xs_onehot


def convert_to_ordinal(
        xs: ArrayLike,
) -> np.ndarray:
    """Convert a one-hot array into an ordinal array.

    The last axis is assumed to be the one-hot dimension.
    """
    return np.argmax(xs, axis=-1)


def split_sample(*arrays, random_state=0, shuffle=True):
    """Split the inputs and labels into two halves for cross-fitting.

    :param arrays: a list of numpy arrays with the same number or rows
    :param random_state: int, RandomState, or None
    :param shuffle: bool
    :return: (arrays0, arrays1)
    """
    split_data = train_test_split(
        *arrays,
        test_size=0.5,
        random_state=random_state,
        shuffle=shuffle,
    )
    return split_data[::2], split_data[1::2]
