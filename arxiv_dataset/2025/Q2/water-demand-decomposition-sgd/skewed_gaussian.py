"""
----------------------------------------------------------------------
Project: water-demand-decomposition-sgd
Filename: skewed_gaussian.py
Author: Roy Elkayam
Created: 2025-05-23
Description:
    This module contains functions to compute skewed Gaussian functions.
    The skewed Gaussian function is defined as:
        f(x) = A * exp(-(x - mu)^2 / (2 * sigma^2)) * (1 + erf(alpha * (x - mu) / (sqrt(2) * sigma)))
    where:
        A: Amplitude
        mu: Mean
        sigma: Standard deviation
        alpha: Skewness parameter
    The function can be used to model asymmetric distributions.
----------------------------------------------------------------------
"""

import numpy as np
from scipy.special import erf

# --- Define Skewed Gaussian Function ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    """Skewed Gaussian function."""
    norm_gaussian = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew


# Define a function to compute individual Gaussian components
def single_gaussian(x, mu, amplitude, sigma, alpha):
    """Compute a single asymmetric Gaussian component."""
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] <= mu:
            y[i] = amplitude * np.exp(-((x[i] - mu) ** 2) / (2 * sigma ** 2))
        else:
            y[i] = amplitude * np.exp(-((x[i] - mu) ** 2) / (2 * (sigma * alpha) ** 2))
    return y
