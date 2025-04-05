from ISCFB.Inexact_FB import inexact_fb
from ISCFB.utils import (div, grad, zero, projection_conj_TV,
                         poly, lin)
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy.linalg import norm
from scipy.signal import convolve2d
# data generation

dataset = "boat"
Z = plt.imread("./ISCFB/data_TV/boat.tiff")
Z = np.array(Z, dtype=np.float)
blur = np.ones((5, 5))/25
Y = convolve2d(Z, blur, mode='same')

s = 0.1*np.mean(Z)
np.random.seed(7)
noise = s * np.random.randn(Z.shape[0], Z.shape[1])
Y += noise


x0 = np.zeros(Y.shape)

L = 1
true_m = 0.01*L
l_reg = 1


def f(x):
    return 0.5*norm(convolve2d(x, blur, mode='same')-Y, ord='fro')**2


def gradf(x):
    return convolve2d(convolve2d(x, blur, mode='same') - Y, blur, mode='same')


def g(x):
    (gx, gy) = grad(x)
    return (l_reg*np.sum(np.sqrt(gx**2+gy**2))
            + 0.5*true_m*norm(x, ord='fro')**2)


def proxg(Y, gradf_y, stepsize, maxiter_inner,
          sigma, zeta, eps, m):
    X0 = (Y-stepsize*gradf_y)/(1+stepsize*m)
    stepsize_mu = stepsize/(1+m*stepsize)
    p_old = np.zeros((X0.shape[0], X0.shape[1], 2))
    u = np.zeros(p_old.shape)
    t = 1
    m_effect = true_m - m
    for i in range(maxiter_inner):
        (gdx, gdy) = grad(-div(u[:, :, 0], u[:, :, 1]) - X0)
        p = projection_conj_TV(u - 1/8 * np.stack([gdx, gdy], axis=2),
                               stepsize_mu*l_reg)
        t_ = (1+sqrt(1+4*t**2))/2
        u = p + (t-1)/t_*(p-p_old)
        t = t_
        p_old = p.copy()

        X = (X0 + div(p[:, :, 0], p[:, :, 1]))/(1+stepsize_mu*m_effect)
        f_primal = (0.5*norm(X-X0, ord='fro')**2
                    + stepsize_mu*(g(X) - 0.5*m*norm(X, ord='fro')**2))
        f_dual = (0.5*norm(X0, ord='fro')**2
                  - 0.5*(1+stepsize_mu*m_effect)*norm(X, ord='fro')**2)

        gap = f_primal - f_dual
        V = (X0-X)/stepsize_mu + m*X
        tol = (0.5*sigma**2*norm(X-Y, ord='fro')**2
               + 0.5*zeta**2*stepsize**2*norm(V + gradf_y, ord='fro')**2
               + eps/2)/(1+stepsize*m)**2
        if gap < max(tol, 1e-16):
            return (X, V, i+1, gap)
    return (X, V, i+1, gap)


maxiter = 1000
maxiter_inner = 1000

mf = f(x0) + g(x0)

sigma = 0.8
zeta = 0.0
xi_name = "zero"


fact = norm(Y, ord='fro')
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

plt.imshow(x_b, cmap='gray')
plt.show()

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
