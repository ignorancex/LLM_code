from __future__ import print_function, division
import numpy as np
from sage.all import StandardTableaux, SemistandardTableaux, Partitions
from math import sqrt
import sys

__all__ = [
    "F_std",
    "F_std_coeff",
    "p_EPR",
    "print_det_std",
    "print_prob_epr",
    "specht",
    "weyl",
    "box_added",
    "GUE0_lambda_max",
    "GUE0_lambda_max_estimate",
]


### Preliminaries ###
def memoize(f):
    memo = {}

    def g(*args):
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]

    return g


def specht(mu):
    """Dimension of Specht module [mu]. Denoted d_mu in [CLM+18]."""
    return StandardTableaux(mu).cardinality().n()


def weyl(d, mu):
    """Dimension of Weyl module V_mu^d. Denoted m_{d,mu} in [CLM+18]."""
    return SemistandardTableaux(shape=mu, max_entry=d).cardinality().n()


def box_added(alpha, d):
    """Partitions of maximal length d obtained by adding one box to alpha."""
    for i, j in alpha.addable_cells():
        if i < d:
            yield alpha.add_cell(i)


### Deterministic PBT ###
def F_std(d, N):
    """Exact formula for the entanglement fidelity from [SSMH17]. Denoted F^std_d(N) in [CLM+18]."""
    # memoize specht() and weyl() results (but only for current call)
    specht_mem, weyl_mem = memoize(specht), memoize(weyl)

    return sum(
        d ** (-N - 2)
        * sum(sqrt(specht_mem(mu) * weyl_mem(d, mu)) for mu in box_added(alpha, d)) ** 2
        for alpha in Partitions(n=N - 1, max_length=d)
    )


def F_std_coeff(d):
    """In [CLM+18], we prove that F^std_d(N) = 1 - c/N + o(1/N), where c = (d^2 - 1) / 4. Return the coefficient c."""
    return (d ** 2 - 1) / 4


### Probabilistic PBT ###
def p_EPR(d, N):
    """
    Exact expression for the success probability from [SSMH17]. Denoted p^EPR_d(N) in [CLM+18].
    We use the formula derived in the proof of Theorem 1.3 of [CLM+18].
    """
    return sum(
        d ** -N * (weyl(d, alpha) * specht(alpha) * N) / (alpha[0] + d)
        for alpha in Partitions(n=N - 1, max_length=d)
    )


def GUE0_lambda_max(d):
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    A = (A + A.T.conj()) / 2
    A = A - A.trace() / d * np.eye(d)
    return np.sort(np.linalg.eigvalsh(A))[-1]


def GUE0_lambda_max_estimate(d, num_samples):
    """
    In [CLM+18], we prove that p^EPR_d(N) = 1 - c*sqrt(d/(N-1)) + o(1/sqrt(N)), where c = E[lambda_max(G)].
    This function estimates the coefficient c.
    """
    return np.mean([GUE0_lambda_max(d) for _ in range(num_samples)])


### Pretty Printing ###
def print_det_std(d, N_min, N_max, N_step=10):
    print("N F O")
    sys.stdout.flush()
    for N in range(N_min, N_max + 1, N_step):
        F = F_std(d, N)
        O = N * (1 - F)
        print(N, F, O)
        sys.stdout.flush()


def print_prob_epr(d, N_min, N_max, N_step=10):
    print("N p FO")
    sys.stdout.flush()
    for N in range(N_min, N_max + 1, N_step):
        p = p_EPR(d, N)
        FO = np.sqrt((N - 1) / d) * (1 - p)
        print(N, p, FO)
        sys.stdout.flush()
