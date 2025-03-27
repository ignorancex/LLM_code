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
import eda_levy_flight as eda_lf
import eda_misc_functions as eda_mf

import estimate_stochastic_process as esp

from matplotlib import rcParams # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Get theoretical histogram data for Brownian motion or Levy flight paths ----
def get_histogram_fitting(
    x,
    t,
    alpha,
    beta,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    hist_sp,
    geometric_flag=False,
    levy_flag=False,
    significance_a=0.01,
    significance_b=-4
):
    """Get theoretical histogram data (probability density function) from
    multiple simulations made for a Brownian motion or Levy flight for a
    specific time t

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
        Stochastic drift of Brownian motion or Levy flight
    sigma : float
        Difussion coefficient of Brownian motion or Levy flight
    x0 : float
        Initial condition of Brownian motion path
    t0 : float
        Initial time for Brownian motion path
    x_threshold : float
        Threshold value for the support of the probability density function
    hist_sp : float or numpy array dtype float
        The values of the histogram of simulated data of restricted Brownian
        motion or restricted Levy flight
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion or Levy flight
        sample as geometric Brownian motion or geometric Levy flight
    levy_flag : bool
        Flag for the selection of Brownian motion or Levy flight sample
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
        transient state of restricted Brownian motion or Levy flight
    params_transient : numpy array dtype float
        Optimal values for the parameters t, alpha, beta, mu, sigma with bounds
    """

    if levy_flag == True:
        hist_transient, params_transient = eda_lf.get_histogram_levy_flight(
            x = x,
            t = t,
            alpha = alpha,
            beta = beta,
            mu = mu,
            sigma = sigma,
            x0 = x0,
            t0 = t0,
            x_threshold = x_threshold,
            hist_rlf = hist_sp,
            geometric_flag = geometric_flag,
            significance_a = significance_a,
            significance_b = significance_b
        )
    else:
        hist_transient, params_transient = eda_lf.get_histogram_brownian_motion(
            x = x,
            t = t,
            mu = mu,
            sigma = sigma,
            x0 = x0,
            t0 = t0,
            x_threshold = x_threshold,
            hist_rbm = hist_sp,
            geometric_flag = geometric_flag,
            significance_a = significance_a,
            significance_b = significance_b
        )

    return hist_transient, params_transient

# Plot final Exploratory Data Analysis for restricted Brownian motion (RBM) or Levy flight (RLF) paths ----
def plot_pdf(
    df_sp,
    ts,
    alpha,
    beta,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    geometric_flag=False,
    levy_flag=False,
    bins=10,
    density=True,
    significance_a=0.01,
    significance_b=-4,
    p_norm=1,
    significant_figures=2,
    width=12,
    height=28,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    fig_cols=4,
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
    """Plot Probability Density Function of Stochastic Process for different
    times and compares with theoretical fitting

    Args:
    ---------------------------------------------------------------------------
    df_sp : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a
        stochastic process with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the random variable or stochastic process
    ts : float or numpy array dtype float
        Arbitrary scalar or vector of real values of the same size of x (space)
    alpha : float
        Stability parameter of Levy flight
    beta : float
        Skew parameter of Levy flight
    mu : float
        Stochastic drift of Brownian motion or Levy flight
    sigma : float
        Difussion coefficient of Brownian motion or Levy flight
    x0 : float
        Initial condition of Brownian motion or Levy flight path
    t0 : float
        Initial time for Brownian motion or Levy flight path
    x_threshold : float
        Threshold value for the support of the probability density function
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion or Levy flight
        sample as geometric Brownian motion or geometric Levy flight
    levy_flag : bool
        Flag for the selection of Brownian motion or Levy flight sample
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    density : bool
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (default value True)
    significance_a : float
        Level of statistical significance at the limits used for curve fitting
        (default value 0.01)
    significance_b : int
        Exponent for statistical significance at the limits used for curve
        fitting when values are close to zero (default value -4)
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
    fig_cols : int
        Number of columns in figure (default value 4)
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
    df_data : pandas DataFrame
        DataFrame with the fitted and simulated data
    """
    # Creation of loop
    if isinstance(ts, float) == True:
        ts = [ts]

    df_final = []
    k = 0
    if len(ts) % fig_cols == 0:
        fig_rows = len(ts) // fig_cols
    else:
        fig_rows = 1 + len(ts) // fig_cols

    # Initialize Plot data
    rcParams.update({"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False})
    fig, ax = plt.subplots(fig_rows, fig_cols)
    fig.set_size_inches(w = width, h = height)

    for t in ts:
        i = k // fig_cols # Integer division
        j = k % fig_cols # Residue

        # Histogram data of simulated data
        bins_sp, bins_midpoint_sp, hist_sp = eda_mf.get_histogram_stochastic_process(
            df_sp = df_sp,
            t = t,
            bins = bins,
            density = density
        )
        del(bins_sp)

        # Histogram theoretical data
        hist_transient, params_transient = get_histogram_fitting(
            x = bins_midpoint_sp,
            t = t,
            alpha = alpha,
            beta = beta,
            mu = mu,
            sigma = sigma,
            x0 = x0,
            t0 = t0,
            x_threshold = x_threshold,
            hist_sp = hist_sp,
            geometric_flag = geometric_flag,
            levy_flag = levy_flag,
            significance_a = significance_a,
            significance_b = significance_b
        )

        # Theoretical data fitting from parameters and respective errors
        if levy_flag == True:
            def estimate_pdf_temp(x, t, alpha, beta, mu, sigma, x0, t0, x_threshold):
                return esp.estimate_pdf_rlf(x, t, alpha, beta, mu, sigma, x0, t0, x_threshold, geometric_flag)
            list_params = ["time", "alpha", "beta", "mu", "sigma", "x0", "t0", "threshold"]
        else:
            def estimate_pdf_temp(x, t, mu, sigma, x0, t0, x_threshold):
                return esp.estimate_pdf_rbm(x, t, mu, sigma, x0, t0, x_threshold, geometric_flag)
            list_params = ["time", "mu", "sigma", "x0", "t0", "threshold"]
        
        transient_prome = estimate_pdf_temp(bins_midpoint_sp, *params_transient[0])    
        transient_lower = estimate_pdf_temp(bins_midpoint_sp, *params_transient[1])
        transient_upper = estimate_pdf_temp(bins_midpoint_sp, *params_transient[2])

        # Resume parameters of regressions (params, R squared, Mean Absolute Error (MAE_p))
        df_final.append(
            eda_mf.resume_params(
                params = params_transient,
                params_type = "Transient state {}".format(k + 1),
                params_names = list_params,
                y = hist_sp,
                y_fitted = transient_prome,
                p_norm = p_norm,
                significant_figures = significant_figures
            )
        )

        # Plot data
        ax[i, j].hist(
            df_sp[df_sp["time"] == t]["value"],
            bins = bins,
            alpha = 0.19,
            facecolor = "blue",
            edgecolor = "darkblue",
            density = density,
            histtype = "stepfilled",
            cumulative = False,
            label = "Simulated data"
        )

        ax[i, j].scatter(bins_midpoint_sp, hist_sp, color = "darkblue", label = "Simulated data")
        ax[i, j].plot(bins_midpoint_sp, hist_transient, color = "red", label = "Transient")
        ax[i, j].plot(bins_midpoint_sp, transient_prome, color = "orange", label = "Transient fitting")

        ax[i, j].fill_between(
            bins_midpoint_sp,
            transient_lower,
            transient_upper,
            where = (
                (transient_upper >= transient_lower) &
                (transient_upper >= transient_prome) &
                (transient_prome >= transient_lower)
            ),
            alpha = 0.19,
            facecolor = "orange",
            interpolate = True,
            label = "Theory fitting"
        )

        ax[i, j].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
        ax[i, j].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
        ax[i, j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        ax[i, j].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))
        ax[i, j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax[i, j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        ax[i, j].tick_params(axis = "x", labelrotation = 90)
        ax[i, j].set_xlabel("$x$", fontsize = fontsize_labels)        
        ax[i, j].set_ylabel("$\Psi(x,t)$", fontsize = fontsize_labels)
        if levy_flag == True:
            ax[i, j].set_yscale("log")
            if geometric_flag == False:
                ax[i, j].set_xscale("symlog")
        ax[i, j].set_title(
            r"({}) $t={}$".format(chr(j + 65), t),
            loc = "left",
            y = 1.005,
            fontsize = fontsize_labels
        )
        ax[i, j].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)

        k += 1
    
    df_final = pd.concat(df_final)
    
    plt.plot()
    fig.tight_layout()
    if save_figures:
        fig.savefig(
            "{}/pdf_{}_{}.png".format(
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

# Simple Plot of final Exploratory Data Analysis for restricted Brownian motion (RBM) or Levy flight (RLF) paths ----
def plot_simple_pdf(
    df_sp,
    ts,
    alpha,
    beta,
    mu,
    sigma,
    x0,
    t0,
    x_threshold,
    geometric_flag=False,
    levy_flag=False,
    bins=10,
    density=True,
    p_norm=1,
    significant_figures=2,
    width=12,
    height=28,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    fig_cols=4,
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
    """Plot Probability Density Function of Stochastic Process for different
    times and compares with theoretical fitting

    Args:
    ---------------------------------------------------------------------------
    df_sp : pandas DataFrame
        DataFrame with the data of the multiple simulations made for a
        stochastic process with three columns namely:
            simulation: the number assigned to the simulation 
            time: computational time of the simulation
            value: value of the random variable or stochastic process
    ts : float or numpy array dtype float
        Arbitrary scalar or vector of real values of the same size of x (space)
    alpha : float
        Stability parameter of Levy flight
    beta : float
        Skew parameter of Levy flight
    mu : float
        Stochastic drift of Brownian motion or Levy flight
    sigma : float
        Difussion coefficient of Brownian motion or Levy flight
    x0 : float
        Initial condition of Brownian motion or Levy flight path
    t0 : float
        Initial time for Brownian motion or Levy flight path
    x_threshold : float
        Threshold value for the support of the probability density function
    geometric_flag : bool
        Flag for the stochastic integration of Brownian motion or Levy flight
        sample as geometric Brownian motion or geometric Levy flight
    levy_flag : bool
        Flag for the selection of Brownian motion or Levy flight sample
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    density : bool
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (default value True)
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
    fig_cols : int
        Number of columns in figure (default value 4)
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
    df_data : pandas DataFrame
        DataFrame with the fitted and simulated data
    """
    # Creation of loop
    if isinstance(ts, float) == True:
        ts = [ts]

    df_final = []
    k = 0
    if len(ts) % fig_cols == 0:
        fig_rows = len(ts) // fig_cols
    else:
        fig_rows = 1 + len(ts) // fig_cols

    # Initialize Plot data
    rcParams.update({"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False})
    fig, ax = plt.subplots(fig_rows, fig_cols)
    fig.set_size_inches(w = width, h = height)

    for t in ts:
        i = k // fig_cols # Integer division
        j = k % fig_cols # Residue

        # Histogram data of simulated data
        bins_sp, bins_midpoint_sp, hist_sp = eda_mf.get_histogram_stochastic_process(
            df_sp = df_sp,
            t = t,
            bins = bins,
            density = density
        )
        del(bins_sp)

        # Histogram theoretical data
        if levy_flag == True:
            hist_transient = esp.estimate_pdf_rlf(
                x = bins_midpoint_sp,
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
            list_params = ["time", "alpha", "beta", "mu", "sigma", "x0", "t0", "threshold"]
            params = [t, alpha, beta, mu, sigma, x0, t0, x_threshold]
        else:
            hist_transient = esp.estimate_pdf_rbm(
                x = bins_midpoint_sp,
                t = t,
                mu = mu,
                sigma = sigma,
                x0 = x0,
                t0 = t0,
                x_threshold = x_threshold,
                geometric_flag = geometric_flag
            )
            list_params = ["time", "mu", "sigma", "x0", "t0", "threshold"]
            params = [t, mu, sigma, x0, t0, x_threshold]

        # Estimation of R squared
        r2_params = mf.estimate_coefficient_of_determination(y = hist_sp, y_fitted = hist_transient)
        r2_params = round(max(0, r2_params) * 100, significant_figures)

        # Estimation of Mean Absolute Error (MAE_p)
        ae_params = mf.estimate_p_norm(x = hist_sp, y = hist_transient, p = p_norm)
        ae_params = round(ae_params, 8)

        # Resume parameters of regressions (params, R squared, Mean Absolute Error (MAE_p))
        df_final.append(
            pd.DataFrame(
                {
                    "fitting" : np.repeat("Transient state {}".format(k + 1), len(list_params)),
                    "params_name" : list_params,
                    "params_value" : params,
                    "r_squared" : np.repeat(r2_params, len(list_params)),
                    "p_norm" : np.repeat(p_norm, len(list_params)),
                    "mae_p" : np.repeat(ae_params, len(list_params))
                }
            )
        )

        # Plot data
        ax[i, j].hist(
            df_sp[df_sp["time"] == t]["value"],
            bins = bins,
            alpha = 0.19,
            facecolor = "blue",
            edgecolor = "darkblue",
            density = density,
            histtype = "stepfilled",
            cumulative = False,
            label = "Simulated data"
        )

        ax[i, j].scatter(bins_midpoint_sp, hist_sp, color = "darkblue", label = "Simulated data")
        ax[i, j].plot(bins_midpoint_sp, hist_transient, color = "red", label = "Theoretical Transient")

        ax[i, j].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
        ax[i, j].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
        ax[i, j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        ax[i, j].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))
        ax[i, j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax[i, j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        ax[i, j].tick_params(axis = "x", labelrotation = 90)
        ax[i, j].set_xlabel(r"$x$", fontsize = fontsize_labels + 1)       
        if levy_flag == True:
            ax[i, j].set_yscale("log")
            if geometric_flag == False:
                ax[i, j].set_ylabel(r"$\Psi_{{LF}}(x,t)$", fontsize = fontsize_labels + 2)
                ax[i, j].set_xscale("symlog", subs = [2, 3, 4, 5, 6, 7, 8, 9])
            else:
                ax[i, j].set_ylabel(r"$\Psi_{{GLF}}(x,t)$", fontsize = fontsize_labels + 2)
                ax[i, j].set_xscale("log")
        else:
            if geometric_flag == False:
                ax[i, j].set_ylabel(r"$\Psi_{{BM}}(x,t)$", fontsize = fontsize_labels + 2)
            else:
                ax[i, j].set_ylabel(r"$\Psi_{{GBM}}(x,t)$", fontsize = fontsize_labels + 2)

        ax[i, j].set_title(
            r"({}) $t={}$, $R^{{2}}={}\%$, $MAE_{{p}}=$ {}".format(
                chr(k + 65),
                t,
                r2_params,
                mf.define_sci_notation_latex(number = ae_params, significant_figures = 5)
            ),
            loc = "left",
            y = 1.005,
            fontsize = fontsize_labels
        )
        ax[i, j].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)

        k += 1
    
    df_final = pd.concat(df_final)
    
    plt.plot()
    fig.tight_layout()
    if save_figures:
        fig.savefig(
            "{}/pdf_simple_{}_{}.png".format(
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
