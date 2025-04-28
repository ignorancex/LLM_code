import numpy as np
from sdpgw.fast_cost_tensor import cost_tensor


def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)


def cost_tensor_numpy(D1, D2):
    # very fast operation, but only works when D1 and D2 are square matrices of
    # the same size. This operation has high memory cost nevertheless
    m, n = D1.shape[0], D2.shape[0]
    dt = D1.dtype
    Lkron = np.kron(D1, np.ones((m, n)).astype(dt)) - np.kron(np.ones((n, m)).astype(dt), D2)
    return (Lkron * Lkron).reshape(m, n, m, n)


def gw_from_cost(D1, D2, pi):
    # calculation from definition of GW
    L = cost_tensor(D1, D2)
    T = np.einsum('ijkl,kl->ij', L, pi)
    return np.sum(T * pi)


def fgw_from_cost(M, D1, D2, pi, alpha):
    # calculation from definition of GW
    L = cost_tensor(D1, D2)
    m, n = pi.shape
    T = np.einsum('ijkl,kl->ij', L, pi)
    return alpha * np.sum(T * pi) + (1 - alpha) * np.sum(M * pi)


def optimality_gap(D1, D2, pi_sdp, pi_star):
    return gw_from_cost(D1, D2, pi_sdp) / gw_from_cost(D1, D2, pi_star)


def approximation_gap(D1, D2, P, pi_sdp):
    m, n = pi_sdp.shape
    L = cost_tensor(D1, D2).reshape(m * n, m * n)
    return gw_from_cost(D1, D2, pi_sdp) / np.sum(L * P)


def sinkhorn_scaling(A, max_iters=1e3, tol=1e-12):
    """Takes a nonnegative NxN matrix A and normalises it to become M so that
    the sum of each row and the sum of each column in M is unity.
    """
    iter = 0
    c = 1. / np.sum(A, 0)
    r = 1. / (A @ c)
    while iter < max_iters:
        iter += 1
        if np.max(np.abs(r @ A * c - 1)) <= tol:
            break
        c = 1. / (r @ A)
        r = 1. / (A @ c)
    return A * np.outer(r, c) / A.shape[0]
