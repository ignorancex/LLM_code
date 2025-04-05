import numpy as np
from math import sqrt
from numba import njit


def zero(k):
    return 0


def poly(k, C):
    def f(x):
        return C/(x+1)**k
    return f


def lin(rho, C):
    def f(x):
        return C*rho**x
    return f


def div(Px, Py):
    n = Px.shape[0]
    m = Py.shape[1]
    fx = Px-Px[np.hstack([0, np.arange(n-1)]), :]
    fx[0, :] = Px[0, :]  # boundary
    fx[-1, :] = -Px[-2, :]

    fy = Py-Py[:, np.hstack([0, np.arange(m-1)])]
    fy[:, 0] = Py[:, 0]  # boundary
    fy[:, -1] = -Py[:, -2]
    return fx+fy


def grad(M):
    (n, m) = M.shape
    fx = M[np.hstack([np.arange(1, n), n-1]), :] - M
    fy = M[:, np.hstack([np.arange(1, m), m-1])] - M
    return (fx, fy)


def projection_conj_TV(x, l):
    return x/(np.maximum(1, np.sqrt(np.sum(x**2, 2))/l)[:, :, None])


@njit
def projection_row(x, l):
    n = x.shape[0]
    for i in range(n):
        nrm = np.linalg.norm(x[i, :])
        if nrm > 0:
            x[i, :] = x[i, :] / nrm * min(nrm, l)
    return x


@njit
def projection_col(x, l):
    n = x.shape[1]
    for i in range(n):
        nrm = np.linalg.norm(x[:, i])
        if nrm > 0:
            x[:, i] = x[:, i] / nrm * min(nrm, l)
    return x


@njit
def ST_vec(x, u):
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


def FistaLasso(X, y, x0, L, maxiter, l_reg, m=0):
    x = x0.copy()
    z = x0.copy()
    t = 1
    if m > 0:
        beta = (1-sqrt(m/(L+m)))/(1+sqrt(m/(L+m)))
    res = np.array([0.5*np.sum((X@x0-y)**2) + l_reg*np.sum(abs(x0))
                    + m/2*np.sum(x0**2)])
    for k in range(maxiter):
        t_ = (1+sqrt(1+4*t**2))/2
        if m == 0:
            beta = (t-1)/t_
        t = t_
        x_ = ST_vec(z - 1/(L+m)*(X.T@(X@z-y)+m*z), l_reg/(L+m))
        z = x_ + beta*(x_-x)
        x = x_.copy()
        f = 0.5*np.sum((X@x-y)**2) + l_reg*np.sum(abs(x)) + m/2*np.sum(x**2)
        res = np.vstack([res, f])
    return (x, res)
