# -*- coding: utf-8 -*-
"""
Created on Sat April 22 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import misc_functions as mf

from scipy.optimize import curve_fit # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Get histogram data of multiple simulations of stochastic process ----
def get_histogram_stochastic_process(df_sp, t, bins=10, density=True):
    """Get histogram data from multiple simulations made for a stochastic
    process (SP) for a specific time t

    Args:
    ---------------------------------------------------------------------------
    df_sp : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a
        stochastic process with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the random variable or stochastic process
    t : float
        Time in which the information will be filtered
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    density : bool
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (default value True)
    
    Returns:
    ---------------------------------------------------------------------------
    bins_sp : float or numpy array dtype float
        Return the bin edges with size of len(hist_sp) + 1
    bins_midpoint_sp : float or numpy array dtype float
        Return the bin edges with the same size of hist_sp
    hist_sp : float or numpy array dtype float
        The values of the histogram
    """

    # Histogram data
    hist_sp, bins_sp = np.histogram(
        df_sp[df_sp["time"] == t]["value"],
        bins = bins,
        density = density
    )

    # Midpoint per bin
    bins_midpoint_sp = np.zeros(len(hist_sp))
    for k in np.arange(0, len(hist_sp), 1):
        bins_midpoint_sp[k] = 0.5 * (bins_sp[k] + bins_sp[k+1])

    return bins_sp, bins_midpoint_sp, hist_sp

# Determination of bounds for arbitrary variable ----
def get_bounds(z, alpha=0.01, beta=-4):
    """Determine correct bounds given a level and exponent of statistical
    significance

    Args:
    ---------------------------------------------------------------------------
    z : float
        variable to determine lower and upper bounds
    alpha : float
        Level of statistical significance (default value 0.01)
    beta : int
        Exponent for statistical significance when values are close to zero
        (default value -4)
    
    Returns:
    ---------------------------------------------------------------------------
    z_lower : float
        Lower bound for variable z
    z_upper : float
        Upper bound for variable z
    """

    lower_quantile = 1 - alpha
    upper_quantile = 1 + alpha

    if z < 0:
        z_lower = upper_quantile * z
        z_upper = lower_quantile * z
    elif z == 0:
        z_upper = 1 * 10**beta
        z_lower = -z_upper
    else:
        z_lower = lower_quantile * z
        z_upper = upper_quantile * z

    return z_lower, z_upper

# Determine the error in the adjustment parameters after curve fitting ----
def get_params_error(popt, pcov):
    """Determine the error in the fitting parameters of a curve fitting

    Args:
    ---------------------------------------------------------------------------
    popt : numpy array dtype float
        Optimal values for the adjustment parameters
    pcov : 2-D numpy array dtype float
        The estimated approximate covariance of popt
    
    Returns:
    ---------------------------------------------------------------------------
    popt : numpy array dtype float
        Optimal values for the adjustment parameters
    popt_lower : numpy array dtype float
        Lower bound for the adjustment parameters
    popt_upper : numpy array dtype float
        Upper bound for the adjustment parameters
    """

    params_error = np.sqrt(np.diag(pcov))
    params_error[np.isinf(params_error)] = 0 
    popt_lower = popt - params_error
    popt_upper = popt + params_error

    return popt, popt_lower, popt_upper

# Get diffusion law of multiple simulations of stochastic process ----
def get_diffusion_law_stochastic_process(df_sp):
    """Get diffusion law data from multiple simulations made for a stochastic
    process (SP)

    Args:
    ---------------------------------------------------------------------------
    df_sp : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a
        stochastic process with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the random variable or stochastic process
    
    Returns:
    ---------------------------------------------------------------------------
    diffusion_mean : float or numpy array dtype float
        Stochastic process mean
    diffusion_variance : float or numpy array dtype float
        Stochastic process variance
    params_diffusion : numpy array dtype float
        Optimal values for the parameters of difussion as power-law with
        coefficient and exponent parameters with bounds
    """

    # Diffusion law data
    df_diffusion = (
        df_sp
            .groupby(["time"])["value"]
            .agg([("mean_value", "mean"), ("variance_value", "var")])
            .reset_index()
    )
    diffusion_mean = df_diffusion["mean_value"].values
    diffusion_variance = df_diffusion["variance_value"].values

    # Diffusion law - Curve fitting as power law (usual diffusion approximation)
    popt_diffusion, pcov_diffusion = curve_fit(
        mf.temporal_fluctuation_scaling,
        diffusion_mean,
        diffusion_variance,
        p0 = [1, 1]
    )

    popt_diffusion, lower_diffusion, upper_diffusion = get_params_error(
        popt = popt_diffusion,
        pcov = pcov_diffusion
    )

    params_diffusion = [popt_diffusion, lower_diffusion, upper_diffusion]

    return diffusion_mean, diffusion_variance, params_diffusion

# Get entropy data and entropy production rate of multiple simulations of stochastic process ----
def get_entropy_stochastic_process(df_sp, dt, p=1, ma_window=10):
    """Get entropy data and entropy production rate from multiple simulations
    made for a stochastic process (SP)

    Args:
    ---------------------------------------------------------------------------
    df_sp : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a
        stochastic process with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the random variable or stochastic process
    dt : float
        Infinitesimal time for integration of stochastic process
    p : float
        Exponent for estimation of Renyi entropy (default value 1 for Shannon
        entropy)
    ma_window : int
        Moving average window for smoothing the entropy production rate
        (default value 10)
    
    Returns:
    ---------------------------------------------------------------------------
    t_sp : float or numpy array dtype float
        Vector of times used to estimate the entropy in the stochastic process
    entropy_sp : float or numpy array dtype float
        Vector of the entropies estimated in the stochastic process
    t_midpoint_sp : float or numpy array dtype float
        Vector of midpoint times used to estimate the entropy production rate
        in the stochastic process
    entropy_production_rate : float or numpy array dtype float
        Vector of the entropy production rates (derivative of entropy)
        estimated in the stochastic process
    epr_smooth_data : 2-D numpy array dtype float
        Entropy production rate smoothed data with components:
            [0] : time
            [1] : smoothed data according ma_window
            [2] : lower bound of smoothed data
            [3] : upper bound of smoothed data
    """

    # Entropy data and Entropy production rate data (derivative of entropy)
    df_entropy = (
        df_sp
            .groupby(["time"])["value"]
            .apply(lambda x: mf.estimate_renyi_entropy(x, p))
            .reset_index()
            .rename(columns = {"value" : "entropy"})
    )
    df_entropy["entropy_production_rate"] = df_entropy["entropy"].diff(periods = 1) / dt
    df_entropy["epr_smooth_time"] = df_entropy["time"].rolling(window = ma_window).mean()
    df_entropy["epr_smooth_std"] = df_entropy["entropy_production_rate"].rolling(window = ma_window).std()
    df_entropy["epr_smooth_mean"] = df_entropy["entropy_production_rate"].rolling(window = ma_window).mean()
    df_entropy["epr_smooth_lower"] = df_entropy["epr_smooth_mean"] - df_entropy["epr_smooth_std"]
    df_entropy["epr_smooth_upper"] = df_entropy["epr_smooth_mean"] + df_entropy["epr_smooth_std"]

    t_midpoint_sp = np.zeros(len(df_entropy["time"].values) - 1)
    for k in np.arange(0, len(df_entropy["time"].values) - 1, 1):
        t_midpoint_sp[k] = 0.5 * (df_entropy["time"].values[k] + df_entropy["time"].values[k+1])
    
    df_entropy = df_entropy[df_entropy["time"] != df_entropy["time"].min()]

    # Final data
    t_sp = df_entropy["time"].values
    entropy_sp = df_entropy["entropy"].values
    entropy_production_rate = df_entropy["entropy_production_rate"].values

    df_entropy = df_entropy[~df_entropy["epr_smooth_mean"].isnull()]
    
    epr_time = df_entropy["epr_smooth_time"].values
    epr_mean = df_entropy["epr_smooth_mean"].values
    epr_lower = df_entropy["epr_smooth_lower"].values
    epr_upper = df_entropy["epr_smooth_upper"].values

    epr_smooth_data = [epr_time, epr_mean, epr_lower, epr_upper]

    return t_sp, entropy_sp, t_midpoint_sp, entropy_production_rate, epr_smooth_data

# Resume the adjustment parameters after curve fitting ----
def resume_params(params, params_type, params_names, y, y_fitted, p_norm, significant_figures):
    """Resume the adjustment parameters after curve fitting

    Args:
    ---------------------------------------------------------------------------
    params : numpy array dtype float
        Optimal values for the curve fitting with bounds
    params_type : string
        Name of the information fitted
    params_names : numpy array dtype string
        Names of parameters
    y : float or numpy array dtype float
        Arbitrary vector of real values of the same size of y_fitted
    y_fitted : float or numpy array dtype float
        Arbitrary vector of real values of the same size of y
    p_norm : float
        p norm used in the estimation on mean absolute error MAE_p
    significant_figures : int
        Number of significant figures in labels
    
    Returns:
    ---------------------------------------------------------------------------
    df_params : pandas DataFrame
        DataFrame with the resume of parameters after curve fitting with eight
        columns namely:
            fitting: Name of the information fitted
            params_name: Names of parameters
            params_value: Optimal values for the curve fitting
            params_lower: Lower bound for optimal values for the curve fitting
            params_upper: Upper bound for optimal values for the curve fitting
            r_squared: Coefficient of determination for the curve fitting
            p_norm: p norm used in the estimation on mean absolute error MAE_p
            mae_p: Mean absolute error MAE_p for the curve fitting
    """

    # Estimation of R squared
    r2_params = mf.estimate_coefficient_of_determination(y = y, y_fitted = y_fitted)
    r2_params = round(max(0, r2_params) * 100, significant_figures)

    # Estimation of Mean Absolute Error (MAE_p)
    ae_params = mf.estimate_p_norm(x = y, y = y_fitted, p = p_norm)

    # Dataframe with the resume of parameters
    df_params = pd.DataFrame(
        {
            "fitting" : np.repeat(params_type, len(params[0])),
            "params_name" : params_names,
            "params_value" : params[0],
            "params_lower" : params[1],
            "params_upper" : params[2],
            "r_squared" : np.repeat(r2_params, len(params[0])),
            "p_norm" : np.repeat(p_norm, len(params[0])),
            "mae_p" : np.repeat(ae_params, len(params[0]))
        }
    )

    return df_params
