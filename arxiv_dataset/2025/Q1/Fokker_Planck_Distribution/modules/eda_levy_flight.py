# -*- coding: utf-8 -*-
"""
Created on Sat April 22 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import pandas as pd # type: ignore
import eda_misc_functions as eda_mf
import estimate_stochastic_process as esp

from scipy.optimize import curve_fit # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Get theoretical histogram data for Brownian motion paths ----
def get_histogram_brownian_motion(
    x,
    t,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    hist_rbm,
    geometric_flag=False,
    significance_a=0.01,
    significance_b=-4
):
    """Get theoretical histogram data (probability density function) from
    multiple simulations made for a Brownian motion for a specific time t

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
    hist_rbm : float or numpy array dtype float
        The values of the histogram of simulated data of restricted Brownian
        motion
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion
    significance_a : float
        Level of statistical significance at the limits used for curve fitting
        (default value 0.01)
    significance_b : int
        Exponent for statistical significance at the limits used for curve
        fitting when values are close to zero (default value -4)
    
    Returns:
    ---------------------------------------------------------------------------
    hist_transient : float or numpy array dtype float
        The values of the histogram (probability density function) for
        transient state of restricted Brownian motion
    params_transient : numpy array dtype float
        Optimal values for the parameters t, mu, sigma with bounds
    """

    # Theoretical histogram data (probability density function)
    hist_transient = esp.estimate_pdf_rbm(
        x = x,
        t = t,
        mu = mu,
        sigma = sigma,
        x0 = x0,
        t0 = t0,
        x_threshold = x_threshold,
        geometric_flag = geometric_flag
    )

    # Curve fitting - Adjustment parameters
    def estimate_pdf_temp(x, t, mu, sigma, x0, t0, x_threshold):
        return esp.estimate_pdf_rbm(x, t, mu, sigma, x0, t0, x_threshold, geometric_flag)
    
    # Curve fitting - Bounds
    t_lower, t_upper = eda_mf.get_bounds(z = t, alpha = significance_a, beta = significance_b)
    mu_lower, mu_upper = eda_mf.get_bounds(z = mu, alpha = significance_a, beta = significance_b)
    sigma_lower, sigma_upper = eda_mf.get_bounds(z = sigma, alpha = significance_a, beta = significance_b)
    x0_lower, x0_upper = eda_mf.get_bounds(z = x0, alpha = significance_a, beta = significance_b)
    t0_lower, t0_upper = eda_mf.get_bounds(z = t0, alpha = significance_a, beta = significance_b)
    x_threshold_lower, x_threshold_upper = eda_mf.get_bounds(z = x_threshold, alpha = significance_a, beta = significance_b)

    bounds_transient = (
        [t_lower, mu_lower, sigma_lower, x0_lower, t0_lower, x_threshold_lower],
        [t_upper, mu_upper, sigma_upper, x0_upper, t0_upper, x_threshold_upper]
    )
    
    try:
        # Curve fitting - Adjustment parameters
        popt_transient, pcov_transient = curve_fit(
            estimate_pdf_temp,
            x,
            hist_rbm,
            p0 = [t, mu, sigma, x0, t0, x_threshold],
            bounds = bounds_transient
        )

        # Curve fitting - Adjustment parameters with bounds
        popt_transient, lower_transient, upper_transient = eda_mf.get_params_error(
            popt = popt_transient,
            pcov = pcov_transient
        )
    except:
        popt_transient = [t, mu, sigma, x0, t0, x_threshold]
        lower_transient = [t_lower, mu_lower, sigma_lower, x0_lower, t0_lower, x_threshold_lower]
        upper_transient = [t_upper, mu_upper, sigma_upper, x0_upper, t0_upper, x_threshold_upper]
    
    params_transient = [popt_transient, lower_transient, upper_transient]

    return hist_transient, params_transient

# Get theoretical histogram data for Levy flight paths ----
def get_histogram_levy_flight(
    x,
    t,
    alpha,
    beta,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    hist_rlf,
    geometric_flag=False,
    significance_a=0.01,
    significance_b=-4
):
    """Get theoretical histogram data (probability density function) from
    multiple simulations made for a Levy flight for a specific time t

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values of the same size of t (time)
    t : float or numpy array dtype float
        Arbitrary scalar or vector of real values of the same size of x (space)
    alpha : float
        Stability parameter of Levy flight
    beta : float
        Skew parameter of Levy flight
    mu : float
        Stochastic drift of Levy flight
    sigma : float
        Difussion coefficient of Levy flight
    x0 : float
        Initial condition of Levy flight path
    t0 : float
        Initial time for Levy flight path
    x_threshold : float
        Threshold value for the support of the probability density function
    hist_rlf : float or numpy array dtype float
        The values of the histogram of simulated data of restricted Levy flight
    geometric_flag : bool
        Flag for the stochastic integration of Levy flight sample as geometric
        Levy flight
    significance_a : float
        Level of statistical significance at the limits used for curve fitting
        (default value 0.01)
    significance_b : int
        Exponent for statistical significance at the limits used for curve
        fitting when values are close to zero (default value -4)
    
    Returns:
    ---------------------------------------------------------------------------
    hist_transient : float or numpy array dtype float
        The values of the histogram (probability density function) for
        transient state of restricted Levy flight
    params_transient : numpy array dtype float
        Optimal values for the parameters t, alpha, beta, mu, sigma with bounds
    """

    # Theoretical histogram data (probability density function)
    hist_transient = esp.estimate_pdf_rlf(
        x = x,
        t = t,
        alpha = alpha,
        beta = beta,
        mu = mu,
        sigma = sigma,
        x0 = x0,
        t0 = t0,
        x_threshold = x_threshold,
        geometric_flag = geometric_flag
    )

    # Curve fitting - Adjustment parameters
    def estimate_pdf_temp(x, t, alpha, beta, mu, sigma, x0, t0, x_threshold):
        return esp.estimate_pdf_rlf(x, t, alpha, beta, mu, sigma, x0, t0, x_threshold, geometric_flag)
    
    # Curve fitting - Bounds
    t_lower, t_upper = eda_mf.get_bounds(z = t, alpha = significance_a, beta = significance_b)
    alpha_lower, alpha_upper = eda_mf.get_bounds(z = alpha, alpha = significance_a, beta = significance_b)
    beta_lower, beta_upper = eda_mf.get_bounds(z = beta, alpha = significance_a, beta = significance_b)
    mu_lower, mu_upper = eda_mf.get_bounds(z = mu, alpha = significance_a, beta = significance_b)
    sigma_lower, sigma_upper = eda_mf.get_bounds(z = sigma, alpha = significance_a, beta = significance_b)
    x0_lower, x0_upper = eda_mf.get_bounds(z = x0, alpha = significance_a, beta = significance_b)
    t0_lower, t0_upper = eda_mf.get_bounds(z = t0, alpha = significance_a, beta = significance_b)
    x_threshold_lower, x_threshold_upper = eda_mf.get_bounds(z = x_threshold, alpha = significance_a, beta = significance_b)

    bounds_transient = (
        [t_lower, alpha_lower, beta_lower, mu_lower, sigma_lower, x0_lower, t0_lower, x_threshold_lower],
        [t_upper, alpha_upper, beta_upper, mu_upper, sigma_upper, x0_upper, t0_upper, x_threshold_upper]
    )

    try:
        # Curve fitting - Adjustment parameters
        popt_transient, pcov_transient = curve_fit(
            estimate_pdf_temp,
            x,
            hist_rlf,
            p0 = [t, alpha, beta, mu, sigma, x0, t0, x_threshold],
            bounds = bounds_transient
        )

        # Curve fitting - Adjustment parameters with bounds
        popt_transient, lower_transient, upper_transient = eda_mf.get_params_error(
            popt = popt_transient,
            pcov = pcov_transient
        )
    except:
        popt_transient = [t, alpha, beta, mu, sigma, x0, t0, x_threshold]
        lower_transient = [t_lower, alpha_lower, beta_lower, mu_lower, sigma_lower, x0_lower, t0_lower, x_threshold_lower]
        upper_transient = [t_upper, alpha_upper, beta_upper, mu_upper, sigma_upper, x0_upper, t0_upper, x_threshold_upper]
    
    params_transient = [popt_transient, lower_transient, upper_transient]

    return hist_transient, params_transient

