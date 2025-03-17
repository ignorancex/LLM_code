# -*- coding: utf-8 -*-
"""
Created on Sat April 22 2024

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import re
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import misc_functions as mf
import matplotlib.pyplot as plt # type: ignore
import matplotlib.ticker as mtick # type: ignore
import eda_misc_functions as eda_mf
import estimate_brownian_motion as ebm

from scipy.optimize import curve_fit # type: ignore
from matplotlib import rcParams # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Get theoretical entropy and entropy production rate data for the restricted Brownian Motion (RBM) ----
def get_entropy_rbm(
    t_rbm,
    entropy_rbm,
    t_midpoint_rbm,
    epr_rbm,
    mu_guess,
    sigma_guess,
    x0_guess,
    t0_guess,
    x_threshold_guess,
    amplitude_guess,
    entropy_gauge_guess,
    epr_gauge_guess
):
    """Get theoretical entropy and entropy production rate from multiple
    simulations made for a restricted Brownian motion

    Args:
    ---------------------------------------------------------------------------
    t_rbm : float or numpy array dtype float
        The values of times used to estimated the entropy of simulated data of
        restricted Brownian motion
    entropy_rbm : float or numpy array dtype float
        The values of the entropy of simulated data of restricted Brownian
        motion
    t_midpoint_rbm : float or numpy array dtype float
        The values of times used to estimated the entropy production rate of
        simulated data of restricted Brownian motion
    epr_rbm : float or numpy array dtype float
        The values of the entropy production rate of simulated data of
        restricted Brownian motion
    mu_guess : float
        Initial guess for the drift of Brownian motion
    sigma_guess : float
        Initial guess for the diffusion coefficient of Brownian motion
    x0_guess : float
        Initial guess for the initial position of Brownian motion
    t0_guess : float
        Initial guess for the initial time of Brownian motion
    x_threshold_guess : float
        Initial guess for the threshold of Brownian motion
    amplitude_guess : float
        Initial guess for the amplitude of entropy of Brownian motion
    entropy_gauge_guess : float
        Initial guess for the initial entropy of the restricted Brownian motion
        (gauge value of entropy)
    epr_gauge_guess : float
        Initial guess for the initial entropy production rate of the Brownian
        motion (gauge value of entropy production rate)
    
    Returns:
    ---------------------------------------------------------------------------
    params_entropy : numpy array dtype float
        Optimal values for the entropy parameters mu, sigma, x0, t0,
        x_threshold, amplitude, h (gauge value of entropy) with bounds
    params_epr : numpy array dtype float
        Optimal values for the entropy production rate (epr) parameters sigma,
        amplitude, phi, h (gauge value of epr) with bounds
    """

    # Curve fitting - Adjustment parameters
    popt_entropy, pcov_entropy = curve_fit(
        ebm.estimate_shannon_entropy_rbm,
        t_rbm,
        entropy_rbm,
        p0 = [mu_guess, sigma_guess, x0_guess, t0_guess, x_threshold_guess, amplitude_guess, entropy_gauge_guess]
    )

    try:
        popt_epr, pcov_epr = curve_fit(
            ebm.estimate_epr_rbm,
            t_midpoint_rbm,
            epr_rbm,
            p0 = [mu_guess, sigma_guess, x0_guess, t0_guess, x_threshold_guess, amplitude_guess, epr_gauge_guess]
        )

        popt_epr, lower_epr, upper_epr = eda_mf.get_params_error(
            popt = popt_epr,
            pcov = pcov_epr
        )
        
        params_epr = [popt_epr, lower_epr, upper_epr]
    
    except:
        popt_epr = [mu_guess, sigma_guess, x0_guess, t0_guess, x_threshold_guess, amplitude_guess, epr_gauge_guess]
        params_epr = [popt_epr, popt_epr, popt_epr]

    # Curve fitting - Adjustment parameters with bounds
    popt_entropy, lower_entropy, upper_entropy = eda_mf.get_params_error(
        popt = popt_entropy,
        pcov = pcov_entropy
    )

    params_entropy = [popt_entropy, lower_entropy, upper_entropy]

    return params_entropy, params_epr

# Plot entropy for restricted Brownian motion paths ----
def plot_entropy_brownian_motion(
    df_bm,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    n_steps,
    p=1,
    ma_window=10,
    p_norm=1,
    significant_figures=2,
    width=12,
    height=28,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    n_cols=4,
    n_x_breaks=10,
    n_y_breaks=10,
    fancy_legend=False,
    usetex=False,
    dpi=200,
    save_figures=True,
    output_path="../output_files",
    information_name="",
    input_generation_date="2024-04-22"
):
    """Plot exploratory data analysis from multiple simulations made for a
    restricted Brownian motion

    Args:
    ---------------------------------------------------------------------------
    df_bm : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a Brownian
        motion with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the Brownian motion
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
    n_steps : int
        Number of temporal steps used in the simulation of Brownian motion path
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    density : bool
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (default value True)
    alpha : float
        Level of statistical significance (default value 0.01)
    beta : int
        Exponent for statistical significance when values are close to zero
        (default value -4)
    p : float
        Exponent for estimation of Renyi entropy (default value 1 for Shannon
        entropy)
    ma_window : int
        Moving average window for smoothing the entropy production rate
        (default value 10)
    p_norm : float
        p norm used in the estimation on mean absolute error MAE_p (default
        value 1)
    significant_figures : int
        Number of significant figures in labels (default value 2)
    width : int
        Width of final plot (default value 12)
    height : int
        Height of final plot (default value 28)
    fontsize_labels : float
        Font size in axis labels (default value 13.5)
    fontsize_legend : float
        Font size in legend (default value 11.5)
    n_cols : int
        Number of columns in legend (default value 4)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    n_y_breaks : int
        Number of divisions in y-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    usetex : bool
        Use LaTeX for renderized plots (default value False)
    dpi : int
        Dot per inch for output plot (default value 200)
    save_figures : bool
        Save figures flag (default value True)
    output_path : string
        Local path for outputs (default value is "../output_files")
    information_name : string
        Name of the output plot (default value "")
    input_generation_date : string
        Date of generation (control version) (default value "2024-04-22")
        
    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        DataFrame with the resume of parameters after multiple curve fitting
        with eight columns namely:
            fitting: Name of the information fitted
            params_name: Names of parameters
            params_value: Optimal values for the curve fitting
            params_lower: Lower bound for optimal values for the curve fitting
            params_upper: Upper bound for optimal values for the curve fitting
            r_squared: Coefficient of determination for the curve fitting
            p_norm: p norm used in the estimation on mean absolute error MAE_p
            mae_p: Mean absolute error MAE_p for the curve fitting
    """

    # Local parameters
    t_min = df_bm["time"].min()
    t_max = df_bm["time"].max()
    dt = (t_max - t_min) / (n_steps - 1)

    # Entropy and entropy production rate data
    x, t = df_bm["value"].values, df_bm["time"].values
    df_bm["entropy"] = ebm.estimate_pdf_rbm(
        x = x,
        t = t,
        mu = mu,
        sigma = sigma,
        x0 = x0,
        t0 = t0,
        x_threshold = x_threshold
    )

    df_entropy = (
        df_bm
            .groupby(["time"])["entropy"]
            .apply(lambda x: np.nanmean(-np.log(x)))
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
    t_rbm = df_entropy["time"].values
    entropy_rbm = df_entropy["entropy"].values
    t_midpoint_rbm = t_midpoint_sp
    epr_rbm = df_entropy["entropy_production_rate"].values

    df_entropy = df_entropy[~df_entropy["epr_smooth_mean"].isnull()]

    epr_time = df_entropy["epr_smooth_time"].values
    epr_mean = df_entropy["epr_smooth_mean"].values
    epr_lower = df_entropy["epr_smooth_lower"].values
    epr_upper = df_entropy["epr_smooth_upper"].values

    epr_smooth_rbm = [epr_time, epr_mean, epr_lower, epr_upper]
    del(epr_smooth_rbm)

    # Entropy and entropy production rate theoretical data
    params_entropy, params_epr = get_entropy_rbm(
        t_rbm = t_rbm,
        entropy_rbm = entropy_rbm,
        t_midpoint_rbm = t_midpoint_rbm,
        epr_rbm = epr_rbm,
        mu_guess = mu,
        sigma_guess = sigma,
        x0_guess = x0,
        t0_guess = t0,
        x_threshold_guess = x_threshold,
        amplitude_guess = 1 * 10**int(np.log10(np.max(entropy_rbm) - np.min(entropy_rbm)) - 1),
        entropy_gauge_guess = entropy_rbm[0],
        epr_gauge_guess = 0
    )

    # Theoretical data fitting from parameters and respective errors
    entropy_prome = ebm.estimate_shannon_entropy_rbm(
        t = t_rbm,
        mu = params_entropy[0][0],
        sigma = params_entropy[0][1],
        x0 = params_entropy[0][2],
        t0 = params_entropy[0][3],
        x_threshold = params_entropy[0][4],
        amplitude = params_entropy[0][5],
        h = params_entropy[0][6]
    )    
    epr_prome = ebm.estimate_epr_rbm(
        t = t_midpoint_rbm,
        mu = params_epr[0][0],
        sigma = params_epr[0][1],
        x0 = params_epr[0][2],
        t0 = params_epr[0][3],
        x_threshold = params_epr[0][4],
        amplitude = params_entropy[0][5], # Only different parameter for EPR
        h = params_epr[0][6]
    )

    # Mask for correct plot of entropy production rate
    if entropy_rbm[1] - entropy_rbm[0] > 0:
        mask = ((epr_prome > 0) & (t_midpoint_rbm > 0.01))
    else:
        mask = ((epr_prome < 0) & (t_midpoint_rbm > 0.01))

    t_midpoint_rbm_ = t_midpoint_rbm[mask]
    epr_rbm_ = epr_rbm[mask]
    epr_prome = epr_prome[mask]

    # Estimation of R squared and Mean Absolute Error (MAE_p)
    r2_entropy = 1 - np.nansum(np.power(entropy_rbm - entropy_prome, 2)) / np.nansum(np.power(entropy_rbm - np.nanmean(entropy_rbm), 2))
    r2_entropy = round(max(0, r2_entropy) * 100, significant_figures)
    r2_epr = 1 - np.nansum(np.power(epr_rbm_ - epr_prome, 2)) / np.nansum(np.power(epr_rbm_ - np.nanmean(epr_rbm_), 2))
    r2_epr = round(max(0, r2_epr) * 100, significant_figures)
    r2_ = [r2_entropy, r2_epr]

    if p_norm == 0:
        ae_entropy = np.exp(0.5 * np.nanmean(np.log(np.power(np.abs(entropy_rbm - entropy_prome), 2))))
        ae_epr = np.exp(0.5 * np.nanmean(np.log(np.power(np.abs(epr_rbm_ - epr_prome), 2))))
    else:
        ae_entropy = np.power(np.nanmean(np.power(np.abs(entropy_rbm - entropy_prome), p)), 1 / p)
        ae_epr = np.power(np.nanmean(np.power(np.abs(epr_rbm_ - epr_prome), p)), 1 / p)

    ae_ = [ae_entropy, ae_epr]

    # Resume parameters of regressions (params, R squared, Mean Absolute Error (MAE_p))
    df_final = pd.concat(
        [
            eda_mf.resume_params(
                params = params_entropy,
                params_type = "Entropy",
                params_names = ["mu", "sigma", "x0", "t0", "threshold", "amplitude", "entropy gauge"],
                y = entropy_rbm,
                y_fitted = entropy_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            ),
            eda_mf.resume_params(
                params = params_epr,
                params_type = "Entropy production rate",
                params_names = ["mu", "sigma", "x0", "t0", "threshold", "amplitude", "epr gauge"],
                y = epr_rbm_,
                y_fitted = epr_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            )
        ]
    )

    # Plot data
    rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": usetex,
            "pgf.rcfonts": False,
            "text.latex.preamble": r"\usepackage{amsfonts}"
        }
    )
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(w = width, h = height)

    # Entropy plot
    entropy_bm = ebm.estimate_shannon_entropy_bm(t_rbm, sigma, t0, params_entropy[0][5], entropy_rbm[0])
    ax[0].scatter(t_rbm, entropy_rbm, c = "darkblue", s = 8, label = "Simulated data")
    ax[0].plot(t_rbm, entropy_prome, c = "firebrick", lw = 2, label = "Theoretical Entropy")
    ax[0].plot(t_rbm, entropy_bm, c = "darkgreen", lw = 2, label = "Standard BM")
    
    # Entropy production rate plot
    epr_bm = ebm.estimate_epr_bm(t = t_midpoint_rbm_, t0 = t0, amplitude = params_entropy[0][5], h = params_epr[0][6])
    offset = np.min([np.nanmin(epr_bm), np.nanmin(epr_rbm), np.nanmin(epr_prome)])
    ax[1].scatter(t_midpoint_rbm, epr_rbm - offset, c = "darkblue", alpha = 0.99, s = 8, label = "Simulated data")
    ax[1].plot(t_midpoint_rbm_, epr_prome - offset, c = "firebrick", lw = 2, label = "Theoretical EPR")
    ax[1].plot(t_midpoint_rbm_, epr_bm - offset, c = "darkgreen", lw = 2, label = "Standard BM")
    
    # Entropy plot - Other features ----
    titles_ = ["Entropy", "Entropy production rate"]
    titles_x = [r"Time $t$", r"Time $t$"]
    titles_y = [
        r"$\mathbb{H}_{{1}}(\Psi_{{BM}},t)$",
        #r"$\frac{d}{dt}\left[\mathbb{H}_{{1}}(\Psi_{{BM}},t)-\mathbb{H}_{{1}}(\Psi_{{BM}},0)\right]$"
        r"$\frac{d}{dt}\mathbb{H}_{{1}}(\Psi_{{BM}},t)$"
    ]

    for j in [0, 1]:
        ax[j].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
        ax[j].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
        ax[j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        ax[j].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))
        ax[j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax[j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        ax[j].tick_params(axis = "x", labelrotation = 90)
        ax[j].set_xlabel(titles_x[j], fontsize = fontsize_labels + 2)        
        ax[j].set_ylabel(titles_y[j], fontsize = fontsize_labels + 2)
        ax[j].set_xscale(value = "log", subs = [2, 3, 4, 5, 6, 7, 8, 9])
        if j == 1:
            ax[j].set_yscale(value = "log", subs = [2, 3, 4, 5, 6, 7, 8, 9])
        ax[j].set_title(
            r"({}) {}, $R^{{2}}={}\%$, $MAE_{{p}}=$ {}".format(
                chr(j + 65),
                titles_[j],
                r2_[j],
                mf.define_sci_notation_latex(number = ae_[j], significant_figures = significant_figures)
            ),
            loc = "left",
            y = 1.005,
            fontsize = fontsize_labels
        )
        ax[j].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)

    plt.plot()
    fig.tight_layout()
    if save_figures:
        fig.savefig(
            "{}/entropy_{}_{}.png".format(
                output_path,
                information_name,
                re.sub("-", "", input_generation_date)
            ),
            bbox_inches = "tight",
            facecolor = fig.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.close()
    
    return df_final
