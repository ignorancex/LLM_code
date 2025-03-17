#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common helper function:
- ode solvers
- MFEM coefficients and sparse matrices
"""
import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat
from scipy.sparse.linalg import splu
import mfem.ser as mfem


#%% time stepping one step ode solver
def euler_step(f, t0, dt, y0):
    """Euler ODE solver step"""
    return y0 + dt * f(t0, y0)


def midpoint_step(f, t0, dt, y0):
    """Midpoint ODE solver step"""
    k1 = f(t0, y0)
    k2 = f(t0 + 0.5 * dt, y0 + 0.5 * dt * k1)
    return y0 + dt * k2

def rk2_tvd_step(f, t0, dt, y0):
    """
    RK2 TVD ODE solver step

    see: Gottlieb/Shu, 1998, p. 78, doi 10.1090/S0025-5718-98-00913-2
    """
    y1 = y0 + dt * f(t0, y0)
    return 0.5 * y0 + 0.5 * y1 + 0.5 * dt * f(t0 + dt, y1)

def rk3_ssp_step(f, t0, dt, y0):
    """
    RK3 SSP ODE solver step

    see: https://docs.mfem.org/3.2/ode_8cpp_source.html#l00062
    """
    k0 = dt * f(t0, y0)
    y1 = y0 + k0
    k1 = dt * f(t0 + dt, y1)
    y2 = 0.75 * y0 + 0.25 * (y1 + k1)
    k2 = dt * f(t0 + 0.5*dt, y2)
    y3 = (1.0/3.0) * y0 + (2.0/3.0) * (y2 + k2)
    return y3


#%%
def orthogonalize(K, M):
    """
    Modified Gram Schmidt orthogonalization

    K = X @ S,  X.T @ M @ X = I_k

    Parameters
    ----------
    K   matrix n x k (numpy.ndarray)
    M   mass matrix n x n (numpy.ndarray)

    Returns
    -------
    X   matrix n x k (numpy.ndarray)
    S   matrix k x k (numpy.ndarray)
    """
    [V, S] = np.linalg.qr(K)

    # modified Gram Schmidt QR decomposition
    m, n = V.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        R[j, j] = np.sqrt(np.dot(V[:, j], M @ V[:, j]))
        Q[:, j] = V[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j].T @ (M @ V[:, k])
            V[:, k] -= R[j, k] * Q[:, j]
    S = R @ S

    return (Q, S)


#%% Tensor stuff
def tensor_diff(X1, S1, V1, X2, S2, V2):
    """Return diffence of two tensors in tensor format"""
    r1 = S1.shape[0]
    r2 = S2.shape[0]
    r = r1 + r2

    X = np.zeros((X1.shape[0], r))
    X[:, :r1] = X1
    X[:, r1:] = X2

    S = np.zeros((r, r))
    S[:r1, :r1] = S1
    S[r1:, r1:] = -S2

    V = np.zeros((V1.shape[0], r))
    V[:, :r1] = V1
    V[:, r1:] = V2

    return (X, S, V)


def L2_norm(X, S, V, Mx, Mv):
    """
    Compute L2 of tensor with respect to mass matrices

    || X S V.T ||_(L_2)

    Parameters
    ----------
    X   spatial factors nx x k (numpy.ndarray)
    S   core tensor k x k (numpy.ndarray)
    V   spatial factors nv x k (numpy.ndarray)
    Mx  spatial mass matrix nx x nx
    Mv  velocity mass matrix nv x nv

    Returns
    -------
    nrm L2 norm of X S V.T

    """
    Ax = X.T @ (Mx @ X)
    Av = V.T @ (Mv @ V)
    return np.sqrt(np.sum((Ax @ S @ Av) * S))


#%% MFEM Helper
def mfem_sparse_to_csr(A):
    """Convert a MFEM sparse matrix to sciyp.sparse.csr_matrix

    Parameters
    ----------
    A : mfem.SparseMatrix

    Returns
    -------
    scipy.sparse.csr_matrix

    """
    height = A.Height()
    width = A.Width()

    AD = A.GetDataArray().copy()
    AI = A.GetIArray().copy()
    AJ = A.GetJArray().copy()
    A = sparse.csr_matrix((AD, AJ, AI), shape=(height, width))
    return A


class x1(mfem.PyCoefficient):
    """MFEM coefficient returning first component f(x) = x_1"""
    def EvalValue(self, x):
        return x[0]


class x2(mfem.PyCoefficient):
    """MFEM coefficient returning first component f(x) = x_2"""
    def EvalValue(self, x):
        return x[1]


class sum_squared(mfem.PyCoefficient):
    """MFEM coefficient returning squared sum of components"""
    def EvalValue(self, x):
        return x[0]**2 + x[1]**2



