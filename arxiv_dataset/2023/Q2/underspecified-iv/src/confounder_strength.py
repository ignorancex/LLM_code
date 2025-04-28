"""
Computation of confounding strength
Estimation of confounding strength

Code based on Paper "Detecting Non-Causal Artifacts in Multivariate Linear Regression Models", Dominik Janzig and
Bernhard Schölkopf, 2018
"""

import numpy as np
import scipy as sc
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def norm_vec(vec):
    """
    Compute the norm of a vector
    Parameters
    ----------
    vec

    Returns
    -------
    float, norm of a vector
    """
    return np.sqrt(sum(vec ** 2))


def density(lin_map, vec):
    d = vec.shape[0]
    # scale vec to unit norm
    vec /= norm_vec(vec)
    vec_in = np.dot(np.linalg.inv(lin_map), vec)
    stretch_factor = norm_vec(vec_in)

    return 1 / (np.linalg.det(lin_map) * (stretch_factor ** d))


def loglikelihood(theta, cov, vec):
    d, _ = cov.shape
    mat_squared = np.identity(d) + theta * np.linalg.inv(cov)
    mat = sc.linalg.sqrtm(mat_squared)
    return - np.log(density(mat, vec))


def estimate_theta(theta_0, C_xx, b):
    res = minimize(loglikelihood,
                   args=(C_xx, b),
                   x0=theta_0,
                   # method="L-BFGS-B",
                   bounds=[(0, None)])
    theta_opt = res.x
    return theta_opt


def estimate_conf_strength(theta_0, x, y):
    """
    estimate confounding strength based on the optimization of the $\theta$ parameter

    Parameters
    ----------
    theta_0: float
        initial value for estimation
    C_xx: ndarray

    b

    Returns
    -------

    """

    # estimate covariance matrix from data
    C_xx = np.cov(x, rowvar=False)  # xx_center.T@xx_center / (xx_center.shape[0] - 1)

    # estimate confounded beta from data
    b = LinearRegression(fit_intercept=True).fit(x, y).coef_
    d, _ = C_xx.shape
    theta_opt = estimate_theta(theta_0, C_xx, b)
    Tinv = np.matrix.trace(np.linalg.inv(C_xx)) / d
    conf_strength = 1 / (1 + 1 / (0.001 + Tinv * theta_opt)) # addition of 0.001 only for numerical stability

    return conf_strength.item()


def compute_conf_strength(beta, m, e):
    """
    Computation of confounding strength

    Parameters
    ----------
    beta
    b
    m
    e

    Returns
    -------

    """

    # second method to compute confounding strength
    prod = np.linalg.pinv(m.T).T@e
    conf_strength = prod@prod / (beta@beta + prod@prod)

    return conf_strength
