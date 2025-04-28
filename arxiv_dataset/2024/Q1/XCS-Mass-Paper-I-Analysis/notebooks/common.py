from astropy.cosmology import LambdaCDM, WMAP9
from astropy.units import Quantity, UnitConversionError
import pandas as pd
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt
import emcee as em
from scipy.optimize import curve_fit, minimize
from getdist import plots, MCSamples

xcs_cosmo = LambdaCDM(70, 0.3, 0.7)
xxl_cosmo = WMAP9
locuss_cosmo = LambdaCDM(70, 0.3, 0.7)
lovisari_cosmo = LambdaCDM(70, 0.3, 0.7)

xcs3p_colour = 'tab:cyan'
xxlgc100_colour = 'firebrick'
locuss_colour = 'black'


# This function is used for getting axis limits for the the one to one comparison plots later
def find_lims(x_dat, y_dat, buffer=0.1):
    # A buffer of some percent (default 10) is added to the max and min values to make the plot 
    #  more easily readable
    lom = 1 - buffer
    him = 1 + buffer
   
    # Depending on whether the input data has + and - errors, just a standard deviation, or no errors at all - this 
    #  alters how we find the limits
    # Using ndim is the way I should have done the other statements, but I can't be bothered to change it now
    if x_dat.ndim == 1:
        x_lims = [np.nanmin(x_dat[np.where(x_dat > 0)[0]]), np.nanmax(x_dat)]
    elif x_dat.shape[1] == 2:
        # The behaviour is largely the same as above, but for symmetrical errors
        lb = x_dat[:, 0] - x_dat[:, 1]
        x_lims = [np.nanmin(lb[np.where(lb > 0)[0]]), np.nanmax(x_dat[:, 0] + x_dat[:, 1])]
    elif x_dat.shape[1] == 3:
        # In this case upper and lower errors are present
        lb = x_dat[:, 0] - x_dat[:, 1]
        # Make sure that we don't count any NaN values, and don't count any negative values
        #  The lower errors are subtracted from the measurements, and upper values added to them
        #  Then max and mins are found 
        x_lims = [np.nanmin(lb[np.where(lb > 0)[0]]), np.nanmax(x_dat[:, 0] + x_dat[:, 2])]

    if y_dat.ndim == 1:
        y_lims = [np.nanmin(y_dat[np.where(y_dat > 0)[0]]), np.nanmax(y_dat)]
    elif y_dat.shape[1] == 3:
        lb = y_dat[:, 0] - y_dat[:, 1]
        y_lims = [np.nanmin(lb[np.where(lb > 0)[0]]), np.nanmax(y_dat[:, 0] + y_dat[:, 2])]
    elif y_dat.shape[1] == 2:
        lb = y_dat[:, 0] - y_dat[:, 1]
        y_lims = [np.nanmin(lb[np.where(lb > 0)[0]]), np.nanmax(y_dat[:, 0] + y_dat[:, 1])]
    
    # Then find the minimum and maximum values from the min and max x and y data, and multiply by the buffer
    lims = Quantity([lom*min([x_lims[0], y_lims[0]]), him*max([x_lims[1], y_lims[1]])])
    
    # Return the limits for the square like for like comparison plot
    return lims


def direct_comparison_plot(xdat: Union[Quantity, List[Quantity]], ydat: Union[Quantity, List[Quantity]], xlabs, ylabs,
                           samp_names, figsize, xscale: Union[str, list] = 'log', yscale: Union[str, list] = 'log', 
                           buffer=0.1, sublabel_fsize=14, savepath=None, low_lim: Quantity = None, 
                           upp_lim: Quantity = None, dat_colour: str = 'black'):
    
    if len(xdat) != len(ydat):
        raise ValueError("The input data are not the same length")
       
    if not isinstance(xdat, (Quantity, np.ndarray)):
        ncols = len(xdat)
        alph_labels = True
    else:
        ncols = 1
        alph_labels = False
        xdat = [xdat]
        ydat = [ydat]
        xlabs = [xlabs]
        ylabs = [ylabs]
    
    if isinstance(xscale, str):
        xscale = [xscale]*ncols
       
    if isinstance(yscale, str):
        yscale = [yscale]*ncols
    
    if isinstance(samp_names, str):
        samp_names = [samp_names]*ncols
    elif len(samp_names) != len(xdat):
        raise ValueError('The number of sample names should be the same as the number of x-data entries')
    
    fig, ax_arr = plt.subplots(ncols=ncols, figsize=figsize)
    if ncols == 1:
        ax_arr = np.array([ax_arr])
        
    # Setting the y-position and font size of the a, b, and c labels that are added below the subplots
    sublabel_ypos = -0.15

    # Iterating through the array of axes objects, setting up the ticks
    for ax_ind, ax in enumerate(ax_arr):
        # Turning on minor ticks and setting it up so they all point inwards - also turn on ticks on the 
        #  top and right axis lines
        ax.minorticks_on()
        ax.tick_params(which='both', top=True, right=True, direction='in')
        if alph_labels:
            # Add the a, b, c, etc labels below the axes
            ax.text(0.5, sublabel_ypos, s='{})'.format(chr(97+ax_ind)), horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes, fontsize=sublabel_fsize)
    
        # Setting the leftmost axis to be current
        plt.sca(ax)
        # Using the function we defined earlier to find appropriate axis limits
        lims = find_lims(xdat[ax_ind], ydat[ax_ind], buffer=buffer)
        if low_lim is not None:
            lims[0] = low_lim
        if upp_lim is not None:
            lims[1] = upp_lim
        
        lims = lims.value

        # Also using the limits to set up a one to one line
        # Then plotting the temperature comparison points
        plt.plot(lims, lims, linestyle='dashed', color='red', label="1:1")
        
        cur_x = xdat[ax_ind]
        cur_y = ydat[ax_ind]
        cur_name = samp_names[ax_ind]
        if cur_x.ndim == 1:
            plt.plot(cur_x, cur_y, 'x', color=dat_colour, label=cur_name)
        else:
            plt.errorbar(cur_x[:, 0].value, cur_y[:, 0].value, xerr=cur_x[:, 1:].T.value, 
                             yerr=cur_y[:, 1:].T.value, color=dat_colour, fmt="x", capsize=2, label=cur_name)
        
        # Setting axis limits
        plt.xlim(lims)
        plt.ylim(lims)
        
        # Setting the scales
        plt.xscale(xscale[ax_ind])
        plt.yscale(yscale[ax_ind])

        # Labels and legend
        plt.xlabel(xlabs[ax_ind], fontsize=15)
        plt.ylabel(ylabs[ax_ind], fontsize=15)
        plt.legend(loc='best', fontsize=13)

    plt.tight_layout()
    
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    
    
def fit_comp(xdat, ydat, prior_bnds, xlab, ylab, norm=None, num_steps=30000, num_walkers=20, linear=True, 
             view_chains=False, view_corner=False, cut_off=1000, buffer=0.1, xscale='linear', yscale='linear'):
    
    # A numpy random object used for the drawing of random samples 
    rng = np.random.default_rng()
    
    if xdat.unit.is_equivalent(ydat.unit):
        ydat = ydat.to(xdat.unit)
    else:
        raise UnitConversionError('ydat units must be convertible to xdat units')
    
    if norm is None:
        norm = Quantity(1, xdat.unit)
    elif not norm.unit.is_equivalent(xdat.unit):
        raise UnitConversionError('Norm units must be convertible to data units')
    
    norm = norm.to(xdat.unit).value
    
    x_data = xdat[:, 0].value
    if xdat.shape[1] == 3:
        x_errs = np.mean(xdat[:, 1:].value, axis=1)
    elif xdat.shape[1] == 2:
        x_errs = xdat[:, 1].value
    else:
        raise ValueError("xdat must have one or two error columns")
    
    y_data = ydat[:, 0].value
    if ydat.shape[1] == 3:
        y_errs = np.mean(ydat[:, 1:].value, axis=1)
    elif ydat.shape[1] == 2:
        y_errs = ydat[:, 1].value
    else:
        raise ValueError("ydat must have one or two error columns")
    
    if linear:
        model_func = straight_line
        model_name = 'Straight Line'
    else:
        model_func = power_law
        model_name = 'Power Law'
    
    allowed = np.where(np.isfinite(x_data) & np.isfinite(y_data))[0]
    x_data = x_data[allowed]/norm
    y_data = y_data[allowed]/norm
    x_errs = x_errs[allowed]/norm
    y_errs = y_errs[allowed]/norm
    
    lower_bounds = [bnd[0] for bnd in prior_bnds]
    upper_bounds = [bnd[1] for bnd in prior_bnds]
    
    if linear:
        fit_par, fit_cov = curve_fit(straight_line, x_data, y_data, p0=[1, 0], sigma=y_errs,
                                     absolute_sigma=True, bounds=(lower_bounds, upper_bounds))
    
    else:
        fit_par, fit_cov = curve_fit(power_law, x_data, y_data, p0=[1, 0], sigma=y_errs,
                                     absolute_sigma=True, bounds=(lower_bounds, upper_bounds))
        
    # This basically finds the order of magnitude of each parameter, so we know the scale on which we should
    #  randomly perturb
    ml_rand_dev = np.power(10, np.floor(np.log10(np.abs(fit_par))))/0.5
    
    # Then that order of magnitude is multiplied by a value drawn from a standard gaussian, and this is what
    #  we perturb the maximum likelihood values with - so we get random start parameters for all
    #  of our walkers
    pos = fit_par + (ml_rand_dev * np.random.randn(num_walkers, 2))
    
    if linear:
        par_names = ['m', 'c']
        sampler = em.EnsembleSampler(num_walkers, 2, log_prob,
                                     args=(x_data, y_data, y_errs, straight_line, prior_bnds))
    else:
        par_names = ['slope', 'norm']
        sampler = em.EnsembleSampler(num_walkers, 2, log_prob,
                                     args=(x_data, y_data, y_errs, power_law, prior_bnds))
        
    sampler.run_mcmc(pos, num_steps, progress=True)
    acc_frac = np.mean(sampler.acceptance_fraction)
    
    if view_chains:
        fig, axes = plt.subplots(nrows=2, figsize=(12, 4), sharex='col')

        plt.suptitle("{m} Parameter Chains".format(m=model_name), fontsize=14, y=1.02)

        chains = sampler.get_chain(discard=cut_off, flat=False)
        for i in range(2):
    #         cur_unit = model_obj.par_units[i]
    #         if cur_unit == Unit(''):
    #             par_unit_name = ""
    #         else:
    #             par_unit_name = r" $\left[" + cur_unit.to_string("latex").strip("$") + r"\right]$"
            ax = axes[i]
            ax.plot(chains[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(chains))
            ax.set_ylabel(par_names[i], fontsize=13)
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step Number", fontsize=13)
        plt.tight_layout()
        plt.show()
    
    flat_chains = sampler.get_chain(discard=cut_off, flat=True)
    
    if view_corner:
        # Need to remove $ from the labels because getdist adds them itself
        stripped_labels = par_names
#         stripped_labels = [n.replace('$', '') for n in model_obj.par_publication_names]
        
        # Setup the getdist sample object
        gd_samp = MCSamples(samples=flat_chains, names=par_names, labels=stripped_labels)

        # And generate the triangle plot
        g = plots.get_subplot_plotter(width_inch=8)
        g.triangle_plot([gd_samp], filled=True)
        plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', top=True, right=True)
    
    lims = find_lims(xdat, ydat, buffer=buffer).value
    x_vals = np.linspace(*lims, 100)
    
    plt.plot(lims, lims, linestyle='dashed', color='red', label="1:1")
    plt.xlim(lims)
    plt.ylim(lims)
        
#     cur_name = samp_names[ax_ind]
    cur_name = model_name
    plt.errorbar(xdat[:, 0].value, ydat[:, 0].value, xerr=xdat[:, 1:].T.value, 
                 yerr=ydat[:, 1:].T.value, fmt="kx", capsize=2, label=cur_name)
    all_inds = np.arange(flat_chains.shape[0])
    
    chosen_inds = rng.choice(all_inds, 10000)
    par_dists = [flat_chains[:, 0][chosen_inds], flat_chains[:, 1][chosen_inds]]
    
    realisations = model_func(x_vals[..., None]/norm, *par_dists)*norm
    median_model = np.percentile(realisations, 50, axis=1)
    upper_model = np.percentile(realisations, 84.1, axis=1)
    lower_model = np.percentile(realisations, 15.9, axis=1)

    # Plotting the power-law model, as well as the confidence limits
    plt.plot(x_vals, median_model, color='cadetblue')
    plt.fill_between(x_vals, lower_model, upper_model, alpha=0.5, interpolate=True,
                         where=upper_model >= lower_model, facecolor='cadetblue')
    plt.plot(x_vals, lower_model, color='cadetblue', linestyle="dashed")
    plt.plot(x_vals, upper_model, color='cadetblue', linestyle="dashed")
    
    plt.xlabel(xlab, fontsize=14)
    plt.ylabel(ylab, fontsize=14)
    
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.legend(loc='best', fontsize=13)
    plt.tight_layout()
    plt.show()
    
    par_meds = np.percentile(flat_chains, 50, axis=0)
    par_errs_p = np.percentile(flat_chains, 84.1, axis=0) - par_meds
    par_errs_m = par_meds - np.percentile(flat_chains, 15.9, axis=0)
    
    par_meds = par_meds.round(3)
    par_errs_p = par_errs_p.round(3)
    par_errs_m = par_errs_m.round(3)
    
    print('{p}={pv} +{pp} -{pm} [1sig]'.format(p=par_names[0], pv=par_meds[0], pp=par_errs_p[0], pm=par_errs_m[0]))
    print('{p}={pv} +{pp} -{pm} [1sig]'.format(p=par_names[1], pv=par_meds[1], pp=par_errs_p[1], pm=par_errs_m[1]))

    
def log_likelihood(theta: np.ndarray, r: np.ndarray, y: np.ndarray, y_err: np.ndarray, m_func) -> np.ndarray:
    """
    Uses a simple Gaussian likelihood function, returns the logged value.
    :param np.ndarray theta: The knowledge we have (think theta in Bayesian parlance) - gets fed
        into the model we've chosen.
    :param np.ndarray r: The radii at which we have measured profile values.
    :param np.ndarray y: The values we have measured for the profile.
    :param np.ndarray y_err: The uncertainties on the measured profile values.
    :param m_func: The model function that is being fit to.
    :return: The log-likelihood value.
    :rtype: np.ndarray
    """
    # Just in case something goes wrong in the model function
    try:
        lik = -np.sum(np.log(y_err*np.sqrt(2*np.pi)) + (((y - m_func(r, *theta))**2) / (2*y_err**2)))
    except ZeroDivisionError:
        lik = np.NaN
    return lik


def log_uniform_prior(theta: np.ndarray, pr: List) -> float:
    """
    This function acts as a uniform prior. Using the limits for the parameters in the chosen
    model (either user defined or default), the function checks whether the passed theta values
    sit within those limits. If they do then of course probability is 1, so we return the natural
    log (as this is a log prior), otherwise the probability is 0, so return -infinity.
    :param np.ndarray theta: The knowledge we have (think theta in Bayesian parlance) - gets fed
        into the model we've chosen.
    :param List pr: A list of upper and lower limits for the parameters in theta, the limits of the
        uniform, uninformative priors.
    :return: The log prior value.
    :rtype: float
    """
    # Check whether theta values are within limits
    theta_check = [pr[t_ind][0] <= t <= pr[t_ind][1] for t_ind, t in enumerate(theta)]
    # If all parameters are within limits, probability is 1, thus log(p) is 0.
    if all(theta_check):
        ret_val = 0.0
    # Otherwise probability is 0, so log(p) is -inf.
    else:
        ret_val = -np.inf

    return ret_val


def log_prob(theta: np.ndarray, r: np.ndarray, y: np.ndarray, y_err: np.ndarray,
             m_func, pr) -> np.ndarray:
    """
    The combination of the log prior and log likelihood.
    :param np.ndarray theta: The knowledge we have (think theta in Bayesian parlance) - gets fed
        into the model we've chosen.
    :param np.ndarray r: The radii at which we have measured profile values.
    :param np.ndarray y: The values we have measured for the profile.
    :param np.ndarray y_err: The uncertainties on the measured profile values.
    :param m_func: The model function that is being fit to.
    :param List pr: A list of upper and lower limits for the parameters in theta, the limits of the
        uniform, uninformative priors.
    :return: The log probability value.
    :rtype: np.ndarray
    """
    lp = log_uniform_prior(theta, pr)
    if not np.isfinite(lp):
        ret_val = -np.inf
    else:
        ret_val = lp + log_likelihood(theta, r, y, y_err, m_func)

    if np.isnan(ret_val):
        ret_val = -np.inf

    return ret_val


def straight_line(x_values: Union[np.ndarray, float], gradient: float, intercept: float) -> Union[np.ndarray, float]:
    """
    As simple a model as you can get, a straight line. Possible uses include fitting very simple scaling relations.
    :param np.ndarray/float x_values: The x_values to retrieve corresponding y values for.
    :param float gradient: The gradient of the straight line.
    :param float intercept: The intercept of the straight line.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return (gradient * x_values) + intercept


def power_law(x_values: Union[np.ndarray, float], slope: float, norm: float) -> Union[np.ndarray, float]:
    """
    A simple power law model, with slope and normalisation parameters. This is the standard model for fitting cluster
    scaling relations in XGA.
    :param np.ndarray/float x_values: The x_values to retrieve corresponding y values for.
    :param float slope: The slope parameter of the power law.
    :param float norm: The normalisation parameter of the power law.
    :return: The y values corresponding to the input x values.
    :rtype: Union[np.ndarray, float]
    """
    return np.power(x_values, slope) * norm