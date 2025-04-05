import numpy as np
from libsvmdata import fetch_libsvm
from ISCFB.Inexact_FB import inexact_fb
from ISCFB.utils import ST_vec, FistaLasso, zero
import matplotlib.pyplot as plt
from numpy.linalg import norm
# data generation

dataset = "madelon"
# dataset = "rcv1.binary"

X, y = fetch_libsvm(dataset, normalize=True)

X = X.toarray()

L = np.linalg.norm(X, ord=2)**2

true_m = 0.00001*L
l_reg = X.shape[0]*0.0002

x0 = np.zeros(X.shape[1])


def f(x):
    return 0.5*norm(X@x - y)**2


def gradf(x):
    return X.T@(X@x-y)


def g(x):
    return np.sum(l_reg*abs(x) + true_m/2*x**2)


def proxg(y, gradf_y, stepsize, maxiter_inner, sigma, zeta, xi, m):
    res = ST_vec((y-stepsize*gradf_y)/(1+true_m*stepsize),
                 stepsize/(1+true_m*stepsize)*l_reg)
    v = (y-stepsize*gradf_y-res)/stepsize
    return (res, v, 1, 0)


sigma = 0
zeta = 0
xi = zero

maxiter = 10000

(x_no_mu, info_no_mu) = inexact_fb(x0, f, g, gradf, proxg, L, 0, sigma, zeta,
                                   xi, maxiter_tot=maxiter, maxiter_inner=1,
                                   backtrack=0, verbose=1, freq=20)


(x, info) = inexact_fb(x0, f, g, gradf, proxg, L, true_m, sigma, zeta, xi,
                       maxiter_tot=maxiter, maxiter_inner=1, backtrack=0,
                       verbose=1, freq=20, tol_backtrack=1e-13)

(xfista, res_fista) = FistaLasso(X, y, x0, L, maxiter, l_reg, true_m)

mf = min(np.min(res_fista), np.min(info[:, 0]), np.min(info_no_mu[:, 0]))

plt.semilogy(np.arange(len(info_no_mu[:, 0])), info_no_mu[:, 0]-mf,
             label="no mu")
plt.semilogy(np.arange(len(info[:, 0])), info[:, 0]-mf, label="known mu")
plt.semilogy(np.arange(len(res_fista)), res_fista-mf, label="fista")
plt.legend()
plt.show()
# %%
