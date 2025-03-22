from mpmath import mpf
import numpy as np
import pandas as pd

from utils import (epsilon2r,
                   epsilon2scale,
                   adp2epsilon_optim,
                   KLDir,
                   r2adp_gohari,
                   r2epsilon)


class DirichletMechanism:

    def __init__(self, epsilon, prior=1, lambda_=2, Delta_2sq=2, Delta_inf=1):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_2sq = Delta_2sq
        self.Delta_inf = Delta_inf
        self.prior = prior
        self.r = epsilon2r(self.epsilon,
                           self.prior,
                           self.lambda_,
                           self.Delta_2sq,
                           self.Delta_inf)
        self.alpha = self.prior + 4*(self.lambda_-1)*self.r*self.Delta_inf

    def sample(self, x, seed=None):
        r = np.random.default_rng(seed).standard_gamma(self.r*np.array(x)+self.alpha)
        return r / r.sum(-1, keepdims=True)

    def sample_series(self, x, seed=None):
        r = np.random.default_rng(seed).standard_gamma(self.r*np.array(x)+self.alpha)
        r = r / r.sum(-1, keepdims=True)
        return pd.Series(r, index=x.index)

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.r = epsilon2r(self.epsilon,
                           self.prior,
                           self.lambda_,
                           self.Delta_2sq,
                           self.Delta_inf)
        self.alpha = self.prior + (self.lambda_-1)*self.r*self.Delta_inf

    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda
        self.r = epsilon2r(self.epsilon,
                           self.prior,
                           self.lambda_,
                           self.Delta_2sq,
                           self.Delta_inf)
        self.alpha = self.prior + (self.lambda_-1)*self.r*self.Delta_inf

    def set_r(self, r):
        self.r = r
        self.epsilon = r2epsilon(self.r,
                                 self.prior,
                                 self.lambda_,
                                 self.Delta_2sq,
                                 self.Delta_inf)
        self.alpha = self.prior + 4*(self.lambda_-1)*self.r*self.Delta_inf

    def get_alpha(self):
        return self.alpha

    def set_dp_epsilon(self, eps_hat, delta):
        self.epsilon = adp2epsilon_optim(eps_hat, delta)
        self.lambda_ = 1 / self.epsilon + 1
        self.r = epsilon2r(self.epsilon,
                           self.prior,
                           self.lambda_,
                           self.Delta_2sq,
                           self.Delta_inf)
        self.alpha = self.prior + (self.lambda_-1)*self.r*self.Delta_inf


class GaussianMechanism:

    def __init__(self, epsilon, lambda_=2, Delta_2sq=2):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_2sq = Delta_2sq
        self.sigma = np.sqrt(self.lambda_*self.Delta_2sq/(2*self.epsilon))

    def sample(self, x, tol=1e-6, seed=None):
        p = np.clip(x+np.random.default_rng(seed).normal(0, self.sigma, np.shape(x)), a_min=tol, a_max=None)
        p = p/p.sum(-1, keepdims=True)
        return p

    def sample_series(self, x, tol=1e-6, seed=None):
        p = np.clip(x+np.random.default_rng(seed).normal(0, self.sigma, np.shape(x)), a_min=tol, a_max=None)
        p = p/p.sum()
        return p

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.sigma = np.sqrt(self.lambda_*self.Delta_2sq/(2*self.epsilon))

    def get_sigma(self):
        return self.sigma

    def set_dp_epsilon(self, eps_hat, delta):
        self.epsilon = None
        self.lambda_ = None
        self.sigma = np.sqrt(2 * np.log(1.25/delta) * self.Delta_2sq) / eps_hat


class LaplaceMechanism:

    def __init__(self, epsilon, lambda_=2, Delta_1=2, d=None):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_1 = Delta_1
        self.d = d
        self.scale = epsilon2scale(self.epsilon, self.lambda_, self.Delta_1)

    def sample(self, x, tol=1e-6, seed=None):
        p = np.clip(x+np.random.default_rng(seed).laplace(0, self.scale, np.shape(x)), a_min=tol, a_max=None)
        p = p/p.sum(-1, keepdims=True)
        return p

    def sample_series(self, x, tol=1e-6, seed=None):
        p = np.clip(x+np.random.default_rng(seed).laplace(0, self.scale, np.shape(x)), a_min=tol, a_max=None)
        p = p/p.sum()
        return p

    def set_rho(self, new_epsilon):
        self.epsilon = new_epsilon
        self.scale = self.Delta_1*np.sqrt(self.lambda_/(2*self.epsilon))

    def get_scale(self):
        return self.scale

    def set_dp_epsilon(self, eps_hat, delta):
        self.epsilon = None
        self.lambda_ = None
        self.scale = self.Delta_1 / eps_hat
        self.scale *= np.sqrt(np.log(0.5 / delta) / np.log(1 / delta))


class Gohari(DirichletMechanism):

    def __init__(self, epsilon, prior=1, lambda_=2, Delta_2sq=2, Delta_inf=1, delta=1e-5, N=None):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.delta = delta
        self.Delta_inf = Delta_inf
        self.prior = prior
        self.N = N
        self.has_none = False
        self.max_epsilon = 0

    def set_r(self, r, X, delta=1e-5):
        self.r = r
        self.delta = delta
        self.alpha = self.prior + 4*(self.lambda_-1) * self.r * self.Delta_inf

        epsilons = []

        num_cats = X.nunique()
        num_cats_unique = num_cats.unique()

        for d in num_cats_unique:
            epsilon = r2adp_gohari(self.r,
                                   self.prior,
                                   self.lambda_,
                                   self.delta,
                                   self.Delta_inf,
                                   self.N,
                                   d)
            if epsilon is not None:
                epsilons += [float(epsilon)]*((num_cats == d).sum())

        if epsilons:
            self.epsilon = np.mean(epsilons)
        else:
            self.has_none = True

    def sample(self, x, tol=1e-6, seed=None):
        r = np.random.default_rng(seed).standard_gamma(self.r*np.array(x)+self.alpha)
        return r / r.sum(-1, keepdims=True)


class MLECalculator:

    def __init__(self, prior):
        self.prior = prior

    def sample(self, x, seed=None):
        p = x + self.prior
        p = p/p.sum(-1, keepdims=True)
        return p

    def sample_series(self, x, seed=None):
        p = x + self.prior
        p = p/p.sum()
        return p


class TruncatedKL:

    def __init__(self, epsilon, lambda_=2, Delta_1=6):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_1 = Delta_1
        self.exp_epsilon = np.sqrt(8 * self.epsilon / self.lambda_)
        self.r = 0.5 * self.exp_epsilon / self.Delta_1

    def sample(self, x, seed=None):
        self.set_epsilon(10)
        x = x/x.sum(-1, keepdims=True)
        out = np.zeros_like(x)
        reject = np.arange(x.shape[0])
        while reject.size > 0:
            x_sub = x[reject]
            r = np.random.default_rng(seed).standard_gamma(self.r*np.array(x_sub))
            r /= r.sum(-1, keepdims=True)
            mask = KLDir(x_sub, r) < 0.5 * self.Delta_1
            out[reject[mask]] = r[mask]
            reject = reject[~mask]
        return out

    def sample_series(self, x, seed=None):
        r = np.random.default_rng(seed).standard_gamma(self.r*np.array(x))
        r = r / r.sum(-1, keepdims=True)
        return pd.Series(r, index=x.index)

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.exp_epsilon = np.sqrt(8 * self.epsilon / self.lambda_)
        self.r = 0.5 * self.exp_epsilon / self.Delta_1

    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda
        self.exp_epsilon = np.sqrt(8 * self.epsilon / self.lambda_)
        self.r = 0.5 * self.exp_epsilon / self.Delta_1
