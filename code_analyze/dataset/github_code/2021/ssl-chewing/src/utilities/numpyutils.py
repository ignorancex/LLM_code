from typing import Tuple, Optional, List

import numpy as np

from utilities.typingutils import is_typed_list


def is_numpy_1d_vector(x):
    """Check if ``x`` is a numpy ndarray and is a 1-D vector."""
    return isinstance(x, np.ndarray) and len(x.shape) == 1


def is_numpy_2d_vector(x, column_vector: Optional[bool] = True):
    """
    Check if ``x`` is a numpy ndarray and is a 1-2 vector.

    :param x: The variable to check
    :param column_vector: If True, vector should be a column vector, i.e. x.shape=(n,1). If False, vector should be a
                          row vector, i.e. x.shape=(1,n). If None, it can be any of the two.
    """
    if not (isinstance(x, np.ndarray) and len(x.shape) == 2):
        return False

    if column_vector is None:
        return True
    if column_vector:
        return x.shape[1] == 1
    else:
        return x.shape[0] == 1


def is_numpy_vector(x):
    return is_numpy_1d_vector(x) or is_numpy_2d_vector(x, None)


def is_numpy_matrix(x, cols: int = None, rows: int = None):
    if not isinstance(x, np.ndarray):
        return False

    if not len(x.shape) == 2:
        return False

    if rows is not None and x.shape[0] != rows:
        return False

    if cols is not None and x.shape[1] != cols:
        return False

    return True


def ensure_numpy_1d_vector(x: np.ndarray, force: bool = False):
    assert isinstance(x, np.ndarray)
    assert isinstance(force, bool)

    shape: Tuple = x.shape

    if len(shape) == 1:
        return x
    elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
        return x.reshape((x.size,))
    elif force:
        return x.reshape((x.size,))
    else:
        raise ValueError("Cannot ensure 1D vector for matrices, unless force=True")


def ensure_numpy_2d_vector(x: np.ndarray, column_vector: bool = True):
    assert isinstance(x, np.ndarray)
    assert isinstance(column_vector, bool)

    shp: Tuple = x.shape
    dim: int = len(shp)

    if dim == 1:
        if column_vector:
            return x.reshape((shp[0], 1))
        else:
            return x.reshape((1, shp[0]))
    elif dim == 2:
        if column_vector:
            assert shp[1] == 1
        else:
            assert shp[0] == 1
        return x
    else:
        raise ValueError("x.shape has " + str(dim) + ", expected 1 or 2 instead")


def setdiff1d_listint(set1: List[int], set2: List[int]) -> List[int]:
    assert is_typed_list(set1, int)
    assert is_typed_list(set2, int)

    nset1: np.ndarray = np.array(set1)
    nset2: np.ndarray = np.array(set2)

    return np.setdiff1d(nset1, nset2).tolist()
