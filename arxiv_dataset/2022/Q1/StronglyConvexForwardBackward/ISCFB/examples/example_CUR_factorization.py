import numpy as np
from math import sqrt
from libsvmdata import fetch_libsvm
from ISCFB.Inexact_FB import inexact_fb
from ISCFB.utils import (projection_row, poly,
                         projection_col, zero, lin)
import matplotlib.pyplot as plt
from numpy.linalg import norm

# data generation

# dataset = "cpusmall"
dataset = "a1a"

W, _ = fetch_libsvm(dataset, normalize=False)


W = W.toarray()
W -= np.mean(W)

W /= norm(W, ord='fro')


L = norm(W, ord=2)**4

print(L)
WTW = W.T@W

true_m = 0.002 * L
l_col = 0.002*sqrt(W.shape[1]/W.shape[0])
l_row = 0.002

x0 = np.zeros((W.shape[1], W.shape[0]))


def f(x):
    return 0.5*norm(W-W@x@W, ord='fro')**2


def gradf(x):
    return -(WTW - WTW@x@W)@(W.T)


def g(x):
    return (true_m/2*norm(x, ord='fro')**2
            + l_col*np.sum(np.sqrt(np.sum(x**2, axis=0)))
            + l_row*np.sum(np.sqrt(np.sum(x**2, axis=1))))


def proxg(Y, gradf_y, stepsize, maxiter_inner,
          sigma, zeta, eps, m):
    m_effect = true_m-m
    X0 = (Y-stepsize*gradf_y)/(1+stepsize*m)
    stepsize_mu = stepsize/(1+m*stepsize)
    X_row = np.zeros(X0.shape)
    X_col = np.zeros(X0.shape)
    for i in range(maxiter_inner):
        X_row = projection_row(X0-X_col, stepsize_mu*l_row)
        X_col = projection_col(X0-X_row, stepsize_mu*l_col)
        X = (X0-(X_row + X_col))/(1+stepsize_mu*m_effect)
        f_primal = (0.5*norm(X0-X, ord='fro')**2
                    + stepsize_mu*(g(X) - 0.5*m*norm(X, ord='fro')**2))
        f_dual = (0.5*norm(X0, ord='fro')**2
                  - 0.5/(1+stepsize_mu*m_effect)
                  * norm(X0-(X_row + X_col), ord='fro')**2)
        gap = f_primal - f_dual
        V = (X0-X)/stepsize_mu + m*X
        tol = (0.5*sigma**2*norm(X-Y, ord='fro')**2
               + 0.5*zeta**2*stepsize**2*norm(V + gradf_y, ord='fro')**2
               + eps/2)/(1+stepsize*m)**2
        if gap < max(tol, 1e-18):
            return (X, V, i+1, gap)
    return (X, V, i+1, gap)


maxiter = 2000
maxiter_inner = 2000

mf = f(x0) + g(x0)

sigma = 0.8
zeta = 0.0
xi_name = "zero"

fact = 1e-5
if xi_name == "zero":
    xi = zero
if xi_name == "poly2":
    xi = poly(2, fact)
if xi_name == "poly3":
    xi = poly(3, fact)
if xi_name == "poly4":
    xi = poly(4, fact)
if xi_name == "lin":
    xi = lin(1-sqrt(true_m/L), fact)

(x_no_mu, info_no_mu) = inexact_fb(x0, f, g, gradf, proxg, L, 0, sigma,
                                   zeta, xi, maxiter_tot=maxiter,
                                   maxiter_inner=maxiter_inner,
                                   backtrack=0, verbose=1, freq=1)

(x, info) = inexact_fb(x0, f, g, gradf, proxg, L, true_m, sigma, zeta, xi,
                       maxiter_tot=maxiter, maxiter_inner=maxiter_inner,
                       backtrack=0, verbose=1, freq=1)

(x_no_mu_b, info_no_mu_b) = inexact_fb(x0, f, g, gradf, proxg, L, 0, sigma,
                                       zeta, xi, maxiter_tot=maxiter,
                                       maxiter_inner=maxiter_inner,
                                       backtrack=1, verbose=1, freq=1)

(x_b, info_b) = inexact_fb(x0, f, g, gradf, proxg, L, true_m, sigma, zeta,
                           xi, maxiter_tot=maxiter,
                           maxiter_inner=maxiter_inner, backtrack=1,
                           verbose=1, freq=1)

mf = min(mf, np.min(info_no_mu[:, 0]), np.min(info[:, 0]),
         np.min(info_b[:, 0]), np.min(info_no_mu_b[:, 0]))


plt.semilogy(np.arange(len(info_no_mu[:, 0])), info_no_mu[:, 0]-mf,
             label="no mu")
plt.semilogy(np.arange(len(info[:, 0])), info[:, 0]-mf, label="known mu")
plt.semilogy(np.arange(len(info_no_mu_b[:, 0])), info_no_mu_b[:, 0]-mf,
             label="no mu b")
plt.semilogy(np.arange(len(info_b[:, 0])), info_b[:, 0]-mf, label="known mu b")

plt.legend()
plt.show()


plt.semilogy(np.cumsum(info_no_mu[:, 1]), info_no_mu[:, 0]-mf, label="no mu")
plt.semilogy(np.cumsum(info[:, 1]), info[:, 0]-mf, label="known mu")
plt.semilogy(np.cumsum(info_no_mu_b[:, 1]), info_no_mu_b[:, 0]-mf,
             label="no mu b")
plt.semilogy(np.cumsum(info_b[:, 1]), info_b[:, 0]-mf, label="known mu b")

plt.legend()
plt.show()
