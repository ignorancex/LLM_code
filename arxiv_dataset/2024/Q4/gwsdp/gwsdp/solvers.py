import gc

import cvxpy as cp
import numpy as np
from scipy import sparse
from tqdm import tqdm

from gwsdp.fast_cost_tensor import cost_tensor
from gwsdp.utils import (cost_tensor_numpy, gw_from_tensor_cost,
                         sinkhorn_scaling)


def solve_gw_sdp(C1, C2, p, q, solver='SCS', max_iters=10000,
                 verbose=False, tol=1e-5):
    # Problem size
    m = p.size
    n = q.size

    # Flatten the cost tensor
    L = cost_tensor(C1, C2).transpose(1, 0, 3, 2).reshape(m * n, m * n)

    # Convert to sparse matrix if sparsity ratio is smaller than 0.5
    if np.sum(L != 0) / L.size <= 0.5:
        L = sparse.csr_matrix(L)

    # Initialize problem
    Pi = cp.Variable((m, n))  # the transportation matrix
    P = cp.Variable((m * n, m * n))

    M = cp.bmat([
        [P, cp.reshape(cp.vec(Pi), (m * n, 1))],
        [cp.reshape(cp.vec(Pi), (1, m * n)), np.ones((1, 1))]
    ])

    constraints = [
        M >> 0,
        Pi >= 0,
        P >= 0,
        Pi @ np.ones(n) == p,
        Pi.T @ np.ones(m) == q,
    ]

    # marg constraint
    ais = np.tile(np.eye(m), n)
    bjs = np.repeat(np.eye(n), m, axis=1)

    for i in range(m):
        constraints += [
            P @ ais[i] == p[i] * cp.vec(Pi)
        ]

    for j in range(n):
        constraints += [
            P @ bjs[j] == q[j] * cp.vec(Pi)
        ]

    # solve the problem
    prob = cp.Problem(cp.Minimize(cp.trace(L @ P.T)), constraints)
    if solver == 'scs':
        prob.solve(solver=cp.SCS, verbose=verbose,
                   max_iters=max_iters, eps=tol)
    elif solver == 'mosek':
        prob.solve(solver=cp.MOSEK, verbose=verbose,
                   mosek_params={'MSK_IPAR_BI_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_SIM_MAX_ITERATIONS': max_iters, })
    else:
        raise TypeError('Solver is not supported.')

    return Pi.value, P.value, prob.value


def solve_fused_gw_sdp(M, C1, C2, p, q, alpha=0.5, max_iters=10000, tol=1e-5,
                       verbose=False, solver='scs'):
    # Problem size
    m = p.size
    n = q.size

    # Flatten the cost tensor
    L = cost_tensor(C1, C2).transpose(1, 0, 3, 2).reshape(m * n, m * n)

    # Convert to sparse matrix if sparsity ratio is smaller than 0.5
    if np.sum(L != 0) / L.size <= 0.5:
        L = sparse.csr_matrix(L)

    # Initialize problem
    Pi = cp.Variable((m, n))  # the transportation matrix
    P = cp.Variable((m * n, m * n))

    M = cp.bmat([
        [P, cp.reshape(cp.vec(Pi), (m * n, 1))],
        [cp.reshape(cp.vec(Pi), (1, m * n)), np.ones((1, 1))]
    ])

    constraints = [
        M >> 0,
        Pi >= 0,
        P >= 0,
        Pi @ np.ones(n) == p,
        Pi.T @ np.ones(m) == q,
    ]

    # marg constraint
    ais = np.tile(np.eye(m), n)
    bjs = np.repeat(np.eye(n), m, axis=1)

    for i in range(m):
        constraints += [
            P @ ais[i] == p[i] * cp.vec(Pi)
        ]

    for j in range(n):
        constraints += [
            P @ bjs[j] == q[j] * cp.vec(Pi)
        ]

    prob = cp.Problem(cp.Minimize(
        (1 - alpha) * cp.trace(M.T @ Pi) + alpha * cp.trace(L @ P.T)), constraints)
    if solver == 'scs':
        prob.solve(solver=cp.SCS, verbose=verbose,
                   max_iters=max_iters, eps=tol)
    elif solver == 'mosek':
        prob.solve(solver=cp.MOSEK, verbose=verbose,
                   mosek_params={'MSK_IPAR_BI_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_SIM_MAX_ITERATIONS': max_iters,
                                 },
                   )
    else:
        raise TypeError('Solver is not supported.')

    return Pi.value, P.value, prob.value
