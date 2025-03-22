# -*- coding: utf-8 -*-
"""
Created on Sat April 23 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore

from scipy.stats import norm # type: ignore

# Global options ----
warnings.filterwarnings("ignore")

# Estimation of probability density function for the restricted geometric Brownian Motion (RGBM) ----
def estimate_pdf_rgbm(x, t, mu, sigma, x0, t0, x_threshold):
    """Estimation of probability density function of restricted geometric
    Brownian motion:

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
    z_x = (np.log(x) - np.log(x0) - mu * tau) / sigma
    z_v = (np.log(x_threshold) - np.log(x0) - mu * tau) / sigma
    normalization = sigma * norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau)) * x
    z = norm.pdf(x = z_x, loc = 0, scale = np.sqrt(tau)) / normalization

    return z

# Estimation of stationary probability density function for restricted geometric Brownian Motion (RGBM) ----
def estimate_stationary_pdf_rgbm(x, x_threshold, lambda_):
    """Estimation of stationary probability density function of restricted
    geometric Brownian motion

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values
    x_threshold : float
        Threshold value for the support of the probability density function
    lambda_ : float
        Scale parameter defined by the stochastic drift and difussion
        coefficient of geometric Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        stationary probability density function of restricted geometric
        Brownian motion
    """
    z = lambda_ * np.power(x_threshold, lambda_) / np.power(x, lambda_ + 1)    
    return z

# Estimation of stationary cumulative density function for restricted geometric Brownian Motion (RGBM) ----
def estimate_stationary_cdf_rgbm(x, x_threshold, lambda_):
    """Estimation of stationary cumulative density function of restricted
    geometric Brownian motion

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values
    x_threshold : float
        Threshold value for the support of the cumulative density function
    lambda_ : float
        Scale parameter defined by the stochastic drift and difussion
        coefficient of geometric Brownian motion
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        stationary cumulative density function of geometric Brownian motion
    """
    z = 1 - np.power(x_threshold / x, lambda_)
    return z

# Estimation of Shannon entropy for the restricted geometric Brownian Motion (RGBM) ----
def estimate_shannon_entropy_rgbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h):
    """Estimation of Shannon entropy of restricted geometric Brownian motion
    with an amplitude A, a gauge condition of the initial differential entropy
    h, and sigma>0

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x0 : float
        Initial condition of geometric Brownian motion path
    t0 : float
        Initial time for geometric Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial differential entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Shannon differential entropy of restricted geometric Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (np.log(x_threshold) - np.log(x0) - mu * tau) / sigma
    z_s = (np.log(x_threshold) - np.log(x0) - (mu - 2 * np.power(sigma, 2)) * tau) / sigma
    
    factor_1 = np.sqrt(2 * np.pi * np.power(x0 * sigma, 2) * tau)
    factor_2 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_3 = 0.5 * z_s * norm.pdf(x = z_v, loc = 0, scale = np.sqrt(tau)) / (factor_2 * np.sqrt(tau))

    # Shannon entropy
    z = h + amplitude * (0.5 + mu * tau + np.log(factor_1) + np.log(factor_2) + factor_3)

    return z

# Estimation of Shannon entropy for the standard geometric Brownian Motion (GBM) ----
def estimate_shannon_entropy_gbm(t, mu, sigma, x0, t0, amplitude, h):
    """Estimation of Shannon entropy of standard geometric Brownian motion with
    an amplitude A, a gauge condition of the initial differential entropy h,
    and sigma>0

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x0 : float
        Initial condition of geometric Brownian motion path
    t0 : float
        Initial time for geometric Brownian motion path
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial differential entropy
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Shannon differential entropy of standard geometric Brownian motion
    """
    # Local variables
    tau = t - t0
    factor_1 = np.sqrt(2 * np.pi * np.power(x0 * sigma, 2) * tau)
    z = h + amplitude * (0.5 + mu * tau + np.log(factor_1))

    return z

# Estimation of Renyi entropy for the restricted geometric Brownian Motion (RGBM) ----
def estimate_renyi_entropy_rgbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h, p):
    """Estimation of Renyi entropy of restricted geometric Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x0 : float
        Initial condition of geometric Brownian motion path
    t0 : float
        Initial time for geometric Brownian motion path
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
        Renyi differential entropy of restricted geometric Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (x_threshold - x0 - mu * tau) / sigma
    z_s = (np.log(x_threshold) - np.log(x0) - (mu - np.power(sigma, 2)) * tau) / sigma
    
    factor_0 = 0.5 * np.log(p) / (p - 1)
    factor_1 = np.sqrt(2 * np.pi * np.power(x0 * sigma, 2) * tau)
    factor_2 = mu * tau + 0.5 * np.power(sigma, 2) * (1 - p) * tau / p
    factor_3 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_4 = norm.sf(x = z_s*np.sqrt(p)-sigma*tau/np.sqrt(p), loc = 0, scale = np.sqrt(tau))  
    factor_5 = (p * np.log(factor_3) - np.log(factor_4)) / (p - 1)  

    # Shannon entropy
    if p == 1:
        z = estimate_shannon_entropy_rgbm(
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
        z = h + amplitude * ((mu - 0.5 * np.power(sigma, 2)) * tau + np.log(factor_1) + np.log(factor_3))
    
    # Renyi entropy (p > 0)
    else:
        z = h + amplitude * (factor_0 + np.log(factor_1) + factor_2 + factor_5)
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Renyi entropy for the standard geometric Brownian Motion (GBM) ----
def estimate_renyi_entropy_gbm(t, mu, sigma, x0, t0, amplitude, h, p):
    """Estimation of Renyi entropy of standard geometric Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x0 : float
        Initial condition of geometric Brownian motion path
    t0 : float
        Initial time for geometric Brownian motion path
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
        Renyi differential entropy of standard geometric Brownian motion
    """
    # Local variables
    tau = t - t0
    factor_0 = 0.5 * np.log(p) / (p - 1)
    factor_1 = np.sqrt(2 * np.pi * np.power(x0 * sigma, 2) * tau)
    factor_2 = mu * tau + 0.5 * np.power(sigma, 2) * (1 - p) * tau / p

    # Shannon entropy
    if p == 1:
        z = estimate_shannon_entropy_gbm(
            t = t,
            mu = mu,
            sigma = sigma,
            x0 = x0,
            t0 = t0,
            amplitude = amplitude,
            h = h
        )
    
    # Min-entropy (Shannon_entropy - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude * ((mu - 0.5 * np.power(sigma, 2)) * tau + np.log(factor_1))
    
    # Renyi entropy (p > 0)
    else:
        z = h + amplitude * (factor_0 + np.log(factor_1) + factor_2)
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Shannon entropy production rate for the restricted geometric Brownian Motion (RGBM) ----
def estimate_epr_rgbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h=0):
    """Estimation of Shannon entropy production rate of restricted geometric
    Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x0 : float
        Initial condition of geometric Brownian motion path
    t0 : float
        Initial time for geometric Brownian motion path
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
        Differential Shannon entropy production rate of restricted geometric
        Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (np.log(x_threshold) - np.log(x0) - mu * tau) / sigma
    z_1 = (np.log(x_threshold) - np.log(x0) + mu * tau) / sigma
    z_2 = (np.log(x_threshold) - np.log(x0) - (mu - 2 * np.power(sigma, 2)) * tau) / sigma
    z_3 = (np.log(x_threshold) - np.log(x0) + (mu + 2 * np.power(sigma, 2)) * tau) / sigma
    
    factor_1 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_2 = norm.pdf(x = z_v, loc = 0, scale = np.sqrt(tau))
    factor_3 = 0.25 * z_3 * factor_2 / (np.power(tau, 1.5) * factor_1)
    factor_4 = 0.25 * z_2 * z_1 * np.power(factor_2 / (tau * factor_1), 2)
    factor_5 = 0.25 * z_2 * factor_2 * (z_1 * z_v - tau) / (np.power(tau, 2.5) * factor_1)

    # Shannon entropy production rate
    z = h + amplitude * (0.5 / tau + mu + factor_3 - factor_4 + factor_5)

    return z

# Estimation of Shannon entropy production rate for the standard geometric Brownian Motion (GBM) ----
def estimate_epr_gbm(t, mu, t0, amplitude, h=0):
    """Estimation of Shannon entropy production rate of standard geometric
    Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    t0 : float
        Initial time for geometric Brownian motion path
    amplitude : float
        Amplitude to make the differential entropy an extensive quantity
    h : float
        Gauge condition of the initial production rate of differential entropy
        (default value 0)
    
    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        Differential Shannon entropy production rate of standard geometric 
        Brownian motion
    """
    # Shannon entropy production rate
    z = h + amplitude * (0.5 / (t - t0) + mu)
    return z

# Estimation of Renyi entropy production rate for the restricted geometric Brownian Motion (RGBM) ----
def estimate_renyi_epr_rgbm(t, mu, sigma, x0, t0, x_threshold, amplitude, h, p):
    """Estimation of Renyi entropy production rate of restricted geometric Brownian
    motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    x0 : float
        Initial condition of geometric Brownian motion path
    t0 : float
        Initial time for geometric Brownian motion path
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
        Differential Renyi entropy production rate of restricted geometric
        Brownian motion
    """
    # Local variables
    tau = t - t0
    z_v = (np.log(x_threshold) - np.log(x0) - mu * tau) / sigma
    z_1 = (np.log(x_threshold) - np.log(x0) + mu * tau) / sigma
    z_2 = (np.log(x_threshold) - np.log(x0) - (mu - np.power(sigma, 2)) * tau) / sigma
    z_3 = (np.log(x_threshold) - np.log(x0) + (mu - np.power(sigma, 2)) * tau) / sigma
    
    factor_1 = norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))    
    factor_2 = norm.pdf(x = z_v, loc = 0, scale = np.sqrt(tau))
    factor_3 = norm.sf(x = z_2*np.sqrt(p)-sigma*tau/np.sqrt(p), loc = 0, scale = np.sqrt(tau))    
    factor_4 = norm.pdf(x = z_2*np.sqrt(p)-sigma*tau/np.sqrt(p), loc = 0, scale = np.sqrt(tau))
    factor_5 = mu + 0.5 * np.power(sigma, 2) * (1 - p) / p
    factor_6 = 0.5 / ((1 - p) * np.power(tau, 1.5))
    factor_7 = (z_3 * np.sqrt(p) + sigma * tau / np.sqrt(p)) * factor_4 / factor_3
    factor_8 = p * z_1 * factor_2 / factor_1
    factor_9 = factor_6 * (factor_7 - factor_8)

    # Shannon entropy production rate
    if p == 1:
        z = estimate_epr_rgbm(
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
        factor_10 = 2 * mu - np.power(sigma, 2)
        z = h + 0.5 * amplitude * (1 / tau + factor_10 - z_1 * factor_2 / (factor_1 * np.power(tau, 1.5)))
    
    # Renyi entropy production rate (p > 0)
    else:
        z = h + amplitude * (0.5 / tau + factor_5 + factor_9)
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z

# Estimation of Renyi entropy production rate for the standard geometric Brownian Motion (GBM) ----
def estimate_renyi_epr_gbm(t, mu, sigma, t0, amplitude, h, p):
    """Estimation of Renyi entropy production rate of geometric Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t : float or numpy array dtype float
        Arbitrary vector of real values (time)
    mu : float
        Stochastic drift of geometric Brownian motion
    sigma : float
        Difussion coefficient of geometric Brownian motion
    t0 : float
        Initial time for geometric Brownian motion path
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
        Differential Renyi entropy production rate of standard geometric
        Brownian motion
    """
    # Shannon entropy production rate
    if p == 1:
        z = estimate_epr_gbm(t = t, mu = mu, t0 = t0, amplitude = amplitude, h = h)
    
    # Min-entropy production rate (Shannon_entropy production rate - A*ln(e))
    elif np.isinf(p) == True:
        z = h + amplitude * (0.5 / (t - t0) + mu - 0.5 * np.power(sigma, 2))
    
    # Renyi entropy production rate (p > 0)
    else:
        z = h + amplitude * (0.5 / (t - t0) + mu + 0.5 * np.power(sigma, 2) * (1 - p) / p)
    
    # Hartley or max-entropy (p=0) is not defined (ln(p=0)/(p-1) -> -infty)

    return z
