'''
Created on May 6, 2025

@author: bernardo
'''

import numpy as np
import math
from sympy import Symbol, lambdify
from sympy.functions.combinatorial.numbers import bell

def poisson_moments(lam, K,symbolic=False):
    """
    Compute the first K+1 raw moments of a Poisson(λ) random variable using Touchard polynomials,
    and return them as a NumPy array of float64.

    Parameters:
    - lam: float (λ > 0), Poisson rate parameter
    - K: int, compute moments from order 0 to K

    Returns:
    - moments: np.ndarray of shape (K+1,), dtype float64
    """
    if K < 0:
        raise ValueError("K must be ≥ 0")
    
    x = Symbol('x')
    moments_sym = [bell(k, x) for k in range(K + 1)]

    if symbolic:
        return [m.subs(x, lam) for m in moments_sym]
    else:
        # Evaluate numerically
        moments_fn = [lambdify(x, m, modules='numpy') for m in moments_sym]
        return np.array([f(lam) for f in moments_fn], dtype=np.float64)

def normal_moments(mu,sigma,K):
    """
    An auxiliary function that returns the first K+1 non-central moments of the normal distribution. 
    Parameters
    ----------
    mu : mean of normal dist
    sigma : std of normal dist
    K: maximum moment to return
    Returns
    -------
    moments_normal : array of shape K+1, with moments 0,1,...,K
    """
    moments_normal = []
    moments_normal.append(1)
    moments_normal.append(mu)
    sigma_sq = sigma**2
    if K>=2:
        for k in range(2, K+1):
            moments_normal.append(moments_normal[1]*moments_normal[k-1]+(k-1)*sigma_sq*moments_normal[k-2])
    return np.asarray(moments_normal)

def exponential_moments(scale,K):
    """
    An auxiliary function that returns the first K+1 non-central moments of the exponential distribution. 
    Parameters
    ----------
    scale : scale of exponential dist
    K: maximum moment to return
    Returns
    -------
    moments_exp : array of shape K+1, with moments 0,1,...,K
    """
    
    moments_exp = np.empty((K + 1,))
    moments_exp[0] = 1
    
    for k in range(1, K + 1):
        moments_exp[k] = math.factorial(k) * np.power(scale, k)
    
    return moments_exp

