# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# The following function names: eye_like, commutator, tensor, are essential quantum operations with definitions that
# follow from the literature. While, for user readability and adaptability, we have used the same namespace
# as QuTip with these functions, the code within is distinct in all cases, save when the operation is so simple and
# universally defined mathematically that alternative syntax is impractical. 

import itertools
from functools import reduce

import cupyx
import numpy as np

import qcheff.operators as qops
from qcheff import qcheff_config


def eye_like(A):
    """
    Identity operator in sparse format with the same shape as A.

    Parameters
    ----------
    A : array_like
        Operator

    Returns
    -------
    identity_operator : array_like
        Identity operator in sparse format with the same shape as A
    """
    xp = cupyx.scipy.get_array_module(A)

    speyelike = xp.sparse.identity(A.shape[0], dtype=A.dtype, format="csr")
    if xp.sparse.issparse(A):
        return speyelike

    else:
        return speyelike.toarray()


def commutator(A, B, kind="normal"):
    """
    Commutator of two operators.

    Parameters
    ----------
    A : array_like
        First operator
    B : array_like
        Second operator
    kind : str, optional
        Kind of commutator. Can be 'normal' or 'anti' (default is 'normal')

    Returns
    -------
    commutator : array_like
        Commutator of two operators
    """
    if kind == "normal":
        return A @ B - B @ A

    elif kind == "anti":
        return A @ B + B @ A

    else:  # pragma: no cover
        msg = f"Unknown commutator kind :{kind}"
        raise TypeError(msg)


def tensor2(A, B):
    """
    Tensor product of two operators.

    Parameters
    ----------
    A : array_like
        First operator
    B : array_like
        Second operator

    Returns
    -------
    tensor_product : array_like
        Tensor product of two operators
    """
    xpsparse = qcheff_config._device_scipy_backend.sparse

    if xpsparse.issparse(A) and xpsparse.issparse(B):
        return xpsparse.kron(A, B, format="csr")
    else:
        return qcheff_config.device_xp_backend.kron(A, B)


def tensor(*args):
    """
    Tensor product of multiple operators.

    Parameters
    ----------
    *args : array_like
        Operators

    Returns
    -------
    tensor_product : array_like
        Tensor product of multiple operators
    """
    return reduce(tensor2, args)


def embed_operator(
    op,
    pos: int,
    system_dims: tuple[int, ...],
):
    """
    Identity wrapping for the appropriate operator.

    Parameters
    ----------
    op : ndarray
        The operator to be wrapped. Ideally, it should have dimensions=ntrunc
    pos : int
        The position of the operator
    system_dims : tuple[int, ...]
        The dimensions of the system

    Returns
    -------
    wrapped_operator : array_like
        The wrapped operator
    """
    # Fully tensored identity in the Full Hilbert Space.
    wrapped_op = [
        op if idx == pos else qops.eye(dim) for idx, dim in enumerate(system_dims)
    ]
    return tensor(*wrapped_op)


# The following functions are taken from QuTiP. Used to maintain compatibility with
# system composition API in QuTiP.
# Source: https://github.com/qutip/qutip/blob/master/qutip/core/states.py
def state_number_enumerate(dims, excitations=None):  # pragma: no cover
    """
    An iterator that enumerates all the state number tuples (quantum numbers of
    the form (n1, n2, n3, ...)) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]): # doctest: +SKIP
        >>>     print(state) # doctest: +SKIP
        ( 0  0 )
        ( 0  1 )
        ( 1  0 )
        ( 1  1 )

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a QuTiP Qobj.

    excitations : integer, optional
        Restrict state space to states with excitation numbers below or
        equal to this value.

    Returns
    -------
    state_number : tuple
        Successive state number tuples that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if excitations is None:
        # in this case, state numbers are a direct product
        yield from itertools.product(*(range(d) for d in dims))
        return

    # From here on, excitations is not None

    # General idea of algorithm: add excitations one by one in last mode (idx =
    # len(dims)-1), and carry over to the next index when the limit is reached.
    # Keep track of the number of excitations while doing so to avoid having to
    # do explicit sums over the states.
    state = (0,) * len(dims)
    nexc = 0
    while True:
        yield state
        idx = len(dims) - 1
        state = state[:idx] + (state[idx] + 1,)
        nexc += 1
        while nexc > excitations or state[idx] >= dims[idx]:
            # remove all excitations in mode idx, add one in idx-1
            idx -= 1
            if idx < 0:
                return
            nexc -= state[idx + 1] - 1
            state = state[:idx] + (state[idx] + 1, 0) + state[idx + 2 :]


def state_number_index(dims, state):  # pragma: no cover
    """
    Return the index of a quantum state corresponding to state,
    given a system with dimensions given by dims.

    Example:

        >>> state_number_index([2, 2, 2], [1, 1, 0])
        6

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a QuTiP Qobj.

    state : list
        State number array.

    Returns
    -------
    idx : int
        The index of the state given by `state` in standard enumeration
        ordering.

    """
    return np.ravel_multi_index(state, dims)


def state_index_number(dims, index):  # pragma: no cover
    """
    Return a quantum number representation given a state index, for a system
    of composite structure defined by dims.

    Example:

        >>> state_index_number([2, 2, 2], 6)
        [1, 1, 0]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    index : integer
        The index of the state in standard enumeration ordering.

    Returns
    -------
    state : tuple
        The state number tuple corresponding to index `index` in standard
        enumeration ordering.

    """
    return np.unravel_index(index, dims)
