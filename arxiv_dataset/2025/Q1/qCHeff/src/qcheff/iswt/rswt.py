# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Everything about RSWT.
import itertools

import numpy as np
import qutip as qt
import scipy as scp
import scipy.linalg as la


def rswt_nested_commutator(A: np.ndarray, B: np.ndarray, order: int):
    """
    Return the nested commutator (Eq. 18)
    C_(l)(A, B) = [A, C_{l-1}(A, B)]

    """
    if order == 0:
        return B
    else:
        return qt.commutator(A, rswt_nested_commutator(A, B, order - 1))


# def rswt_generator_coupling_step(D: np.ndarray, V: np.ndarray, j: int, k: int):
#     """ """
#     pass


def rswt_create_generator(D: np.ndarray, V: np.ndarray):
    """
    Create the generator from the Hamiltonian.
    """

    assert D.shape == V.shape, "D and V must have the same shape."

    dim = D.shape[0]

    div_mat = np.zeros_like(D)
    # Divide by the correct thinga
    # Matrix which contains the division factors
    # Might be a better way to do this.
    for i, j in itertools.permutations(range(dim), r=2):
        div_mat[i, j] = D[i, i] - D[j, j]

    return V / div_mat


def rswt_step(H: np.ndarray, order: int, n: int):
    """
    Do a single step of RSWT.

    order = K
    """

    m = np.floor(order / np.exp2(n))
    # Replace later.
    D, V = None  # split_diag_offdiag_matrix(H)

    S = rswt_create_generator(D, V)

    return D + np.sum(
        [
            t * rswt_nested_commutator(S, V, order=t) / scp.special.factorial(t + 1)
            for t in range(m)
        ]
    )


def rswt(H: np.ndarray, order: int):
    """
    Order = K
    """

    nmax = np.floor(np.log2(order))

    for n in range(nmax):
        H = rswt_step(H, order, n)

    return H
