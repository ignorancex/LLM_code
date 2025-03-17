# -*- coding: utf-8 -*-
"""
Created on Wednesday August 14 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import misc_functions as mf

from scipy.stats import norm # type: ignore
from scipy.stats import levy_stable # type: ignore
from functools import partial

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Simulate a sample of Brownian motion path ----
def estimate_brownian_motion_sample(
    mu,
    sigma,
    threshold,
    threshold_flag,
    geometric_flag,
    log_path,
    log_filename,
    verbose,
    bm_args_list
):
    """Estimation of a Brownian motion sample according to:
        x0           = bm_args_list[0]
        t0           = bm_args_list[1]
        tf           = bm_args_list[2]
        n_steps      = bm_args_list[3]
        n_simulation = bm_args_list[4]

    Args
    ---------------------------------------------------------------------------
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    threshold : float
        Threshold value for the restriction of Brownian motion sample
    threshold_flag : bool
        Flag for the restriction of Brownian motion sample values
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_bm")
    verbose : int
        Provides additional details as to what the computer is doing when
        sample of Brownian motion is running
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    tf : float
        Last time for Brownian motion path
    n_steps : int
        Number of temporal steps used in the simulation of Brownian motion path
    n_simulation : int
        Number of simulation given to the Brownian motion path

    Returns
    ---------------------------------------------------------------------------
    df_bm : pandas DataFrame
        Brownian motion path
    """

    # Definition of Brownian motion parameters
    x0 = bm_args_list[0]
    t0 = bm_args_list[1]
    tf = bm_args_list[2]
    n_steps = bm_args_list[3]
    n_simulation = bm_args_list[4]

    # Define the initial condition and time interval elapsed
    t = np.linspace(start = t0, stop = tf, num = n_steps)
    z = np.zeros(n_steps)
    z[0] = x0
    
    # Parameters for integration of Stochastic differential equation
    dt = (tf - t0) / n_steps # Temporal step
    dW = np.random.normal(loc = 0, scale = np.sqrt(dt), size = n_steps) # Noise
    
    # Integration of stochastic differential equation
    if geometric_flag == True:
        for j in range(0, n_steps - 1):
            z[j + 1] = z[j] + mu * z[j] * dt + sigma * z[j] * dW[j]
            if threshold_flag == True:
                if z[j + 1] < threshold:
                    z[j + 1] = threshold
    else:
        for j in range(0, n_steps - 1):
            z[j + 1] = z[j] + mu * dt + sigma * dW[j]
            if threshold_flag == True:
                if z[j + 1] < threshold:
                    z[j + 1] = threshold
        
    # Definition of final dataframe
    df_bm = pd.DataFrame(
        {
            "simulation" : np.repeat(n_simulation, n_steps),
            "restricted" : np.repeat(threshold_flag, n_steps),
            "time" : t,
            "value" : z
        }
    )

    # Function development
    if verbose >= 1:
        process = "BM"
        if geometric_flag == True:
            process = "GBM"
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write(
                "{}: restricted={}, mu={}, sigma={}, x0={}, t0={}, tf={}, n_s={}, sim={}\n".format(
                    process,
                    threshold_flag,
                    mu,
                    sigma,
                    x0,
                    t0,
                    tf,
                    n_steps,
                    n_simulation
                )
            )

    return df_bm

# Simulate a sample of Levy flight path ----
def estimate_levy_flight_sample(
    alpha,
    beta,
    mu,
    sigma,
    threshold,
    threshold_flag,
    geometric_flag,
    log_path,
    log_filename,
    verbose,
    lf_args_list
):
    """Estimation of a Levy flight sample according to:
        x0           = lf_args_list[0]
        t0           = lf_args_list[1]
        tf           = lf_args_list[2]
        n_steps      = lf_args_list[3]
        n_simulation = lf_args_list[4]

    Args
    ---------------------------------------------------------------------------
    alpha : float
        Stability parameter of Levy flight
    beta : float
        Skew parameter of Levy flight
    mu : float
        Stochastic drift of Levy flight
    sigma : float
        Difussion coefficient of Levy flight
    threshold : float
        Threshold value for the restriction of Levy flight sample
    threshold_flag : bool
        Flag for the restriction of Levy flight sample values
    geometric_flag : bool
        Flag for the stochastic integration of Levy flight sample as
        geometric Levy flight
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_lf")
    verbose : int
        Provides additional details as to what the computer is doing when
        sample of Levy flight is running
    x0 : float
        Initial condition of Levy flight path
    t0 : float
        Initial time for Levy flight path
    tf : float
        Last time for Levy flight path
    n_steps : int
        Number of temporal steps used in the simulation of Levy flight path
    n_simulation : int
        Number of simulation given to the Levy flight path

    Returns
    ---------------------------------------------------------------------------
    df_lf : pandas DataFrame
        Levy flight path
    """

    # Definition of Levy flight parameters
    x0 = lf_args_list[0]
    t0 = lf_args_list[1]
    tf = lf_args_list[2]
    n_steps = lf_args_list[3]
    n_simulation = lf_args_list[4]

    # Define the initial condition and time interval elapsed
    t = np.linspace(start = t0, stop = tf, num = n_steps)
    z = np.zeros(n_steps)
    z[0] = x0
    
    # Parameters for integration of Stochastic differential equation
    dt = (tf - t0) / n_steps # Temporal step
    dW = levy_stable.rvs(
        alpha = alpha,
        beta = beta,
        loc = 0,
        scale = np.power(dt, 1 / alpha),
        size = n_steps
    ) # Noise
    
    # Integration of stochastic differential equation
    if geometric_flag == True:
        for j in range(0, n_steps - 1):
            z[j + 1] = z[j] + mu * z[j] * dt + sigma * z[j] * dW[j]
            if threshold_flag == True:
                if z[j + 1] < threshold:
                    z[j + 1] = threshold
    else:
        for j in range(0, n_steps - 1):
            z[j + 1] = z[j] + mu * dt + sigma * dW[j]
            if threshold_flag == True:
                if z[j + 1] < threshold:
                    z[j + 1] = threshold
    
    # Definition of final dataframe
    df_lf = pd.DataFrame(
        {
            "simulation" : np.repeat(n_simulation, n_steps),
            "restricted" : np.repeat(threshold_flag, n_steps),
            "time" : t,
            "value" : z
        }
    )

    # Function development
    if verbose >= 1:
        process = "LF"
        if geometric_flag == True:
            process = "GLF"
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write(
                "{}: restricted={}, alpha={}, beta={}, mu={}, sigma={}, x0={}, t0={}, tf={}, n_s={}, sim={}\n".format(
                    process,
                    threshold_flag,
                    alpha,
                    beta,
                    mu,
                    sigma,
                    x0,
                    t0,
                    tf,
                    n_steps,
                    n_simulation
                )
            )

    return df_lf

# Simulation of multiple Brownian motion with the same initial conditions ----
def simulate_brownian_motion(
    mu,
    sigma,
    threshold,
    threshold_flag,
    geometric_flag,
    bm_args_list,
    log_path="../logs",
    log_filename="log_bm",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of multiple Brownian motion samples according to:
        x0           = bm_args_list[k, 0]
        t0           = bm_args_list[k, 1]
        tf           = bm_args_list[k, 2]
        n_steps      = bm_args_list[k, 3]
        n_simulation = bm_args_list[k, 4]
    for k in {1, 2,..., n_samples}

    Args
    ---------------------------------------------------------------------------
    mu : float
        Stochastic drift of Brownian motion
    sigma : float
        Difussion coefficient of Brownian motion
    threshold : float
        Threshold value for the restriction of Brownian motion sample
    threshold_flag : bool
        Flag for the restriction of Brownian motion sample values
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion sample as
        geometric Brownian motion
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    tf : float
        Last time for Brownian motion path
    n_steps : int
        Number of temporal steps used in the simulation of Brownian motion path
    n_samples : int
        Number of simulations for Brownian motion paths
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_bm")
    verbose : int
        Provides additional details as to what the computer is doing when
        multiple Brownian motion samples is running (default value is 1)
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)

    Returns
    ---------------------------------------------------------------------------
    df_bm : pandas DataFrame
        Multiple Brownian motion paths
    """

    # Auxiliary function for simulations of Brownian paths
    fun_local = partial(
        estimate_brownian_motion_sample,
        mu,
        sigma,
        threshold,
        threshold_flag,
        geometric_flag,
        log_path,
        log_filename,
        verbose
    )
    
    # Parallel loop for simulations of Brownian paths
    df_bm = mf.parallel_run(fun = fun_local, arg_list = bm_args_list, tqdm_bar = tqdm_bar)
    df_bm = pd.concat(df_bm)
    
    return df_bm

# Simulation of multiple Levy flight with the same initial conditions ----
def simulate_levy_flight(
    alpha,
    beta,
    mu,
    sigma,
    threshold,
    threshold_flag,
    geometric_flag,
    lf_args_list,
    log_path="../logs",
    log_filename="log_lf",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of multiple Levy flight samples according to:
        x0           = lf_args_list[k, 0]
        t0           = lf_args_list[k, 1]
        tf           = lf_args_list[k, 2]
        n_steps      = lf_args_list[k, 3]
        n_simulation = lf_args_list[k, 4]
    for k in {1, 2,..., n_samples}

    Args
    ---------------------------------------------------------------------------
    alpha : float
        Stability parameter of Levy flight
    beta : float
        Skew parameter of Levy flight
    mu : float
        Stochastic drift of Levy flight
    sigma : float
        Difussion coefficient of Levy flight
    threshold : float
        Threshold value for the restriction of Levy flight sample
    threshold_flag : bool
        Flag for the restriction of Levy flight sample values
    geometric_flag : bool
        Flag for the stochastic integration of Levy flight sample as
        geometric Levy flight
    x0 : float
        Initial condition of Levy flight path
    t0 : float
        Initial time for Levy flight path
    tf : float
        Last time for Levy flight path
    n_steps : int
        Number of temporal steps used in the simulation of Levy flight path
    n_samples : int
        Number of simulations for Levy flight paths
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_lf")
    verbose : int
        Provides additional details as to what the computer is doing when
        multiple Levy flight samples is running (default value is 1)
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)

    Returns
    ---------------------------------------------------------------------------
    df_lf : pandas DataFrame
        Multiple Levy flight paths
    """

    # Auxiliary function for simulations of Levy flight paths
    fun_local = partial(
        estimate_levy_flight_sample,
        alpha,
        beta,
        mu,
        sigma,
        threshold,
        threshold_flag,
        geometric_flag,
        log_path,
        log_filename,
        verbose
    )
    
    # Parallel loop for simulations of Levy flight paths
    df_lf = mf.parallel_run(fun = fun_local, arg_list = lf_args_list, tqdm_bar = tqdm_bar)
    df_lf = pd.concat(df_lf)
    
    return df_lf

# Estimation of probability density function for the restricted Brownian Motion (RBM) ----
def estimate_pdf_rbm(x, t, mu, sigma, x0, t0, x_threshold, geometric_flag=False):
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

    if geometric_flag == True:
        z_x = (np.log(x) - np.log(x0) - mu * tau) / sigma
        z_v = (np.log(x_threshold) - np.log(x0) - mu * tau) / sigma
        normalization = sigma * norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau)) * x
    else:
        z_x = (x - x0 - mu * tau) / sigma
        z_v = (x_threshold - x0 - mu * tau) / sigma
        normalization = sigma * norm.sf(x = z_v, loc = 0, scale = np.sqrt(tau))

    z = norm.pdf(x = z_x, loc = 0, scale = np.sqrt(tau)) / normalization

    return z

# Estimation of probability density function for the restricted Levy flight (RLF) ----
def estimate_pdf_rlf(x, t, alpha, beta, mu, sigma, x0, t0, x_threshold, geometric_flag=False):
    """Estimation of probability density function of restricted Levy flight:

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
    geometric_flag : bool
        Flag for the stochastic integration of Levy flight sample as
        geometric Levy flight

    Returns:
    ---------------------------------------------------------------------------
    z : float or numpy array dtype float
        probability density function of restricted Levy flight
    """

    tau = t - t0

    if geometric_flag == True:
        z_x = (np.log(x) - np.log(x0) - mu * tau) / sigma
        z_v = (np.log(x_threshold) - np.log(x0) - mu * tau) / sigma
        normalization = sigma * levy_stable.sf(
            x = z_v,
            alpha = alpha,
            beta = beta,
            loc = 0,
            scale = np.power(tau, 1 / alpha)
        ) * x
    else:
        z_x = (x - x0 - mu * tau) / sigma
        z_v = (x_threshold - x0 - mu * tau) / sigma
        normalization = sigma * levy_stable.sf(
            x = z_v,
            alpha = alpha,
            beta = beta,
            loc = 0,
            scale = np.power(tau, 1 / alpha)
        )

    z = levy_stable.pdf(
        x = z_x,
        alpha = alpha,
        beta = beta,
        loc = 0,
        scale = np.power(tau, 1 / alpha)
    ) / normalization

    return z
