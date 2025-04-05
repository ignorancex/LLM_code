import numpy as np
from libsvmdata import fetch_libsvm
from numpy.linalg import norm
from ISCFB.Inexact_FB import inexact_fb
from ISCFB.utils import ST_vec, zero
from sklearn.linear_model import ElasticNet
# data generation

dataset = "w5a"
# dataset = "rcv1.binary"

X, y = fetch_libsvm(dataset, normalize=False)

X = X.toarray()

L = norm(X, ord=2)**2/X.shape[0]

true_m = 0.001*L
l_reg = 0.0004

params = [(0, 1e-4), (0.001*L, 4e-4), (0.01*L, 1e-5),
          (1e-3*L, 1e-3), (5e-3*L, 5e-3)]


def f(x):
    return 0.5*norm(X@x - y)**2/X.shape[0]


def gradf(x):
    return X.T@(X@x-y)/X.shape[0]


for (true_m, l_reg) in params:

    x0 = np.zeros(X.shape[1])

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

    maxiter = 5000
    maxiter_inner = 5000

    (x_no_mu, info_no_mu) = inexact_fb(x0, f, g, gradf, proxg, L, 0, sigma,
                                       zeta, xi, maxiter_tot=maxiter,
                                       maxiter_inner=maxiter_inner,
                                       backtrack=1, verbose=0, freq=20)

    (x, info) = inexact_fb(x0, f, g, gradf, proxg, L, true_m, sigma, zeta, xi,
                           maxiter_tot=maxiter, maxiter_inner=maxiter_inner,
                           backtrack=1, verbose=0, freq=20)

    EN = ElasticNet(alpha=(l_reg + true_m),
                    l1_ratio=l_reg/(l_reg + true_m),
                    tol=1e-8, max_iter=30000, fit_intercept=False)

    EN.fit(X, y)

    x_lasso = EN.coef_

    print(np.sum(abs(x_lasso) >= 1e-12))
    f_lasso = f(x_lasso)+g(x_lasso)
    assert(abs(f_lasso - f(x_no_mu)-g(x_no_mu))/abs(f_lasso) <= 1e-7)
    print("No mu passed test.")

    assert(abs(f_lasso - f(x)-g(x))/abs(f_lasso) <= 1e-7)
    print("With mu passed test.")
