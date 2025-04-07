# pylint: disable=C0103
"""Helper function for plotting
"""
from typing import List

import numpy as np
from sympy import Matrix, S
from sympy.core.relational import Relational

import qlp.eqn_converter as qe

DEPENDENTS = [S("x_0"), S("x_1")]


def f(x0: float, x1: float) -> float:
    r"""Hyperbole function

    Implements
    $$
        f(x_0, x_1) = 8 -(x_0 - 2)^2 - (x_1 - 2)^2
    $$
    """
    return 8 - (x0 - 2) ** 2 - (x1 - 2) ** 2


def get_omega_0(n_bits: int, as_numeric: bool = False) -> Matrix:
    r"""Computes the matrix which represents $f$ as a bit matrix-vector operation

    E.g.,
    $$
        f(x_0, x_1) = \psi_\alpha M_{\alpha \beta} \psi_\beta
        \, , \qquad
        M_{\alpha \beta}
        =
        - Q_{0\alpha}Q_{0\beta}
        - Q_{1\alpha}Q_{1\beta}
        + 4Q_{0\alpha}\delta_{\alpha\beta}
        + 4Q_{1\alpha}\delta_{\alpha\beta}
    $$
    where $\psi$ is a bit vector and $Q$ is given by
    $$
    \begin{pmatrix}
        x_0 \\ x_1
    \end{pmatrix}
    =
    Q
    \vec \psi \, ,
    $$

    Arguments:
        n_bits: Number of bits.
        as_numeric: Return numpy array if True. Else sympy expression.
    """
    q = qe.get_bit_map(len(DEPENDENTS), n_bits)
    mat = -q.T @ q + 4 * np.diag(np.sum(q, axis=0))
    return np.array(mat) if as_numeric else mat


def get_omega(
    inequalities: List[Relational], n_bits: int, p: int = 10, as_numeric: bool = False
) -> Matrix:
    """Computes `omega_0 - p * constraint**2` in the bit basis with slack variables.

    Calls `qlp.eqn_converter.constraints_to_matrix` to compute the constrained matrix.

    Arguments:
        inequalities: The list of inequalities.
        n_bits: The number of bits to represent integers.
        p: The penalty term. Must be larger than the maximal value of the opt function.
        as_numeric: Use numpy instead of sympy.
    """
    n_vars = len(DEPENDENTS) + len(inequalities)
    q = qe.get_bit_map(n_vars=n_vars, n_bits=n_bits)
    a, b = qe.constraints_to_matrix(
        inequalities, dependents=DEPENDENTS, as_numeric=as_numeric
    )
    omega = -p * qe.get_constrained_matrix(q, a, b, as_numeric=as_numeric)
    nx = len(DEPENDENTS) * n_bits
    omega[:nx, :nx] += get_omega_0(n_bits, as_numeric=as_numeric)
    return omega
