import numpy as np
from numba import jit

# Least Squares


@jit(nopython=True)
def R_ls(x, x_):
    return 1/2 * (x - x_)**2


@jit(nopython=True)
def I_ls(x, x_):
    return (x-x_)**2

# Logistic Regression


@jit(nopython=True)
def R_lr(x, x_):
    return -x * np.exp(x_) / (1 + np.exp(x_)) + np.log(np.exp(x) + 1)


@jit(nopython=True)
def R_lr_entropy(x, x_):
    return -x_ * np.exp(x_) / (1 + np.exp(x_)) + np.log(np.exp(x_) + 1)


@jit(nopython=True)
def I_lr(x, x_):
    gx = np.exp(x) / (1 + np.exp(x))
    gx_ = np.exp(x_) / (1 + np.exp(x_))
    return (gx - gx_)**2


@jit(nopython=True)
def gauss_hermite_numba(L, xi, w, n, which="R"):
    total = 0.0
    for i in range(n):
        for j in range(n):
            x0 = np.sqrt(2) * (L[0, 0] * xi[i] + L[0, 1] * xi[j])
            x1 = np.sqrt(2) * (L[1, 0] * xi[i] + L[1, 1] * xi[j])
            if which == "R_lr":
                total += w[i] * w[j] * R_lr(x0, x1)
            if which == "R_lr_entropy":
                total += w[i] * w[j] * R_lr_entropy(x0, x1)
            if which == "I_lr":
                total += w[i] * w[j] * I_lr(x0, x1)
            if which == "R_ls":
                total += w[i] * w[j] * R_ls(x0, x1)
            if which == "I_ls":
                total += w[i] * w[j] * I_ls(x0, x1)
    return total / np.pi


def gauss_hermite(B, which, n=50):
    # check if B is PSD
    if not np.all(np.linalg.eigvals(B) > 0):
        L = np.linalg.cholesky(B + 1e-6 * np.eye(2))
    else:
        L = np.linalg.cholesky(B)
    xi, w = np.polynomial.hermite.hermgauss(n)
    return gauss_hermite_numba(L, xi, w, n, which)
