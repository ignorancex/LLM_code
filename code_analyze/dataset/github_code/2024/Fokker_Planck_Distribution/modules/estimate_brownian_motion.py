# -*- coding: utf-8 -*-
"""
Created on Sat April 20 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore

from scipy.stats import norm # type: ignore

# Global options ----
warnings.filterwarnings("ignore")

# Estimation of probability density function for the restricted Brownian Motion (RBM) ----
def estimate_pdf_rbm(x, t, mu, sigma, x0, t0, x_threshold):
    """Estimation of probability density function of restricted Brownian
    motion:

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values of the same size of t (time)
    t : float or numpy array dtype float
        Arbitrary scalar or vector of real values of the same size of x (space)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion

    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        probability density function of restricted Brownian motion
    """
    tau = t - t0
    z_x = (x - x0 - mu * tau) / sigma
    z_v = (x_threshold - x0 - mu * tau) / sigma
    normalization = sigma * norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))
    z = norm.pdf(x = z_x, loc = 0, scale = np.sqrt(tau)) / normalization

    return z

# Estimation of stationary probability density function for the restricted Brownian Motion (RBM) ----
def estimate_stationary_pdf_rbm(x, x_threshold, lambda_):
    r"""Estimation of stationary probability density function of restricted
    Brownian motion as:
    :math:`\lambda\exp{\left(-\lambda(x-x_{th})\right)}`
    where $x\geq x_{th}$ and $\lambda=2\mu\sigma^{-2}>0$.

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values
    x_threshold : float
        Threshold value for the support of the probability density function
    lambda_ : float
        Scale parameter defined by the stochastic drift and difussion
        coefficient of Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        stationary probability density function of restricted Brownian motion
    """
    z = lambda_ * np.exp(-lambda_ * (x - x_threshold))
    return z

# Estimation of Shannon entropy for the restricted Brownian Motion (RBM) ----
def estimate_shannon_entropy_rbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h):
    """Estimation of Shannon entropy of restricted Brownian motion with an
    amplitude A, a gauge condition of the initial differential entropy h, and
    sigma>0

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial differential entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Shannon differential entropy of restricted Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (x_threshold - x0 - mu * tau) / sigma
    
    factor_1 = np.sqrt(2 * np.pi * np.power(sigma, 2) * tau)
    factor_2 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_3 = 0.5 * z_v * norm.pdf(x = z_v, loc = 0, scale = np.sqrt(tau)) / (factor_2 * np.sqrt(tau))

    # Shannon entropy
    z = h + amplitude * (0.5 + np.log(factor_1) + np.log(factor_2) + factor_3)
    return z

# Estimation of Shannon entropy for the standard Brownian Motion (BM) ----
def estimate_shannon_entropy_bm(t, sigma, t0, amplitude, h):
    """Estimation of Shannon entropy of Brownian motion with an amplitude A, a
    gauge condition of the initial differential entropy h, and sigma>0

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    sigma : float
        Difussion coefficient of Brownian motion
    t0 : float
        Initial time for Brownian motion path
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial differential entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Shannon differential entropy of Brownian motion
    """
    # Local variables
    tau = t - t0
    factor_1 = np.sqrt(2 * np.pi * np.power(sigma, 2) * tau)

    # Shannon entropy
    z = h + amplitude * (0.5 + np.log(factor_1))
    return z

# Estimation of Renyi entropy for the restricted Brownian Motion (RBM) ----
def estimate_renyi_entropy_rbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h, p):
    """Estimation of Renyi entropy of restricted Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial differential entropy
    p : float
        Exponent for estimation of Renyi entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Renyi differential entropy of restricted Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (x_threshold - x0 - mu * tau) / sigma
    
    factor_0 = 0.5 * np.log(p) / (p - 1)
    factor_1 = np.sqrt(2 * np.pi * np.power(sigma, 2) * tau)
    factor_2 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_3 = norm.sf(x = z_v*np.sqrt(p), loc = 0, scale = np.sqrt(tau))    

    # Shannon entropy
    if p == 1:
        z = estimate_shannon_entropy_rbm(
            t = t,
            mu = mu,
            sigma = sigma,
            x0 = x0,
            t0 = t0,
            x_threshold = x_threshold,
            amplitude = amplitude,
            h = h
        )
    
    # Min-entropy (Shannon_entropy - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude * (np.log(factor_1) + np.log(factor_2))
    
    # Renyi entropy (p > 0)
    else:
        z = h + amplitude * (factor_0 + np.log(factor_1) + (p * np.log(factor_2) - np.log(factor_3)) / (p - 1))
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Renyi entropy for the standard Brownian Motion (BM) ----
def estimate_renyi_entropy_bm(t, sigma, t0, amplitude, h, p):
    """Estimation of Renyi entropy of standard Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    sigma : float
        Difussion coefficient of Brownian motion
    t0 : float
        Initial time for Brownian motion path
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial differential entropy
    p : float
        Exponent for estimation of Renyi entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Renyi differential entropy of standard Brownian motion
    """
    # Local variables
    tau = t - t0    
    factor_0 = 0.5 * np.log(p) / (p - 1)
    factor_1 = np.sqrt(2 * np.pi * np.power(sigma, 2) * tau)  

    # Shannon entropy
    if p == 1:
        z = estimate_shannon_entropy_bm(t = t, sigma = sigma, t0 = t0, amplitude = amplitude, h = h)

    # Min-entropy (Shannon_entropy - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude * np.log(factor_1)
    
    # Renyi entropy (p > 0)
    else:
        z = h + amplitude * (factor_0 + np.log(factor_1))
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Shannon entropy production rate for the restricted Brownian Motion (RBM) ----
def estimate_epr_rbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h=0):
    """Estimation of Shannon entropy production rate of restricted Brownian
    motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Shannon entropy production rate of restricted Brownian
        motion
    """
    # Local variables
    tau = t - t0
    z_v = (x_threshold - x0 - mu * tau) / sigma
    z_s = (x_threshold - x0 + mu * tau) / sigma
    
    factor_1 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_2 = norm.pdf(x = z_v, loc = 0, scale = np.sqrt(tau))
    factor_3 = 0.25 * z_s * factor_2 / (np.power(tau, 1.5) * factor_1)
    factor_4 = 0.25 * z_v * z_s * np.power(factor_2 / (tau * factor_1), 2)
    factor_5 = 0.25 * z_v * factor_2 * (z_v * z_s - tau) / (np.power(tau, 2.5) * factor_1)

    # Shannon entropy production rate
    z = h + amplitude * (0.5 / tau + factor_3 - factor_4 + factor_5)
    return z

# Estimation of Shannon entropy production rate for the standard Brownian Motion (BM) ----
def estimate_epr_bm(t, t0, amplitude, h=0):
    """Estimation of Shannon entropy production rate of standard Brownian
    motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    t0 : float
        Initial time for Brownian motion path
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Shannon entropy production rate of standard Brownian
        motion
    """
    # Shannon entropy production rate
    z = h + 0.5 * amplitude / (t - t0)
    return z

# Estimation of Renyi entropy production rate for the restricted Brownian Motion (RBM) ----
def estimate_renyi_epr_rbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h, p):
    """Estimation of Renyi entropy production rate of restricted Brownian
    motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    p : float
        Exponent for estimation of Renyi entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Renyi entropy production rate of restricted Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (x_threshold - x0 - mu * tau) / sigma
    z_s = (x_threshold - x0 + mu * tau) / sigma

    factor_1 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_2 = norm.pdf(x = z_v, loc = 0, scale = np.sqrt(tau))
    factor_3 = norm.sf(x = z_v*np.sqrt(p), loc = 0, scale = np.sqrt(tau))    
    factor_4 = norm.pdf(x = z_v*np.sqrt(p), loc = 0, scale = np.sqrt(tau))
    factor_5 = 0.5 * z_s / ((1 - p) * np.power(tau, 1.5))
    factor_6 = np.sqrt(p) * factor_4 / factor_3
    factor_7 = p * factor_2 / factor_1
    factor_8 = factor_5 * (factor_6 - factor_7)

    # Shannon entropy production rate
    if p == 1:
        z = estimate_epr_rbm(
            t = t,
            mu = mu,
            sigma = sigma,
            x0 = x0,
            t0 = t0,
            x_threshold = x_threshold,
            amplitude = amplitude,
            h = h
        )
    
    # Min-entropy production rate (Shannon_entropy production rate - A*ln(e))
    elif np.isinf(p) == True:
        z = h + 0.5 * amplitude * (1 / tau + z_s * factor_2 / (factor_1 * np.power(tau, 1.5)))
    
    # Renyi entropy production rate (p > 0)
    else:
        z = h + amplitude * (0.5 / tau + factor_8)
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Renyi entropy production rate for the standard Brownian Motion (BM) ----
def estimate_renyi_epr_bm(t, t0, amplitude, h):
    """Estimation of Renyi entropy production rate of Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    t0 : float
        Initial time for Brownian motion path
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Renyi entropy production rate of standard Brownian motion
    """
    # Renyi or Shannon or Min-entropy entropy production rate
    z = h + 0.5 * amplitude / (t - t0)

    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z
