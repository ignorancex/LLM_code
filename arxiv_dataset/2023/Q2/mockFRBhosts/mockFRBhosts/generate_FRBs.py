#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides functions to generate an FRB population using frbpoppy.

Functions:
    lognorm_mu(mu, sigma, w_min, shape, z):
        Draws from a lognormal distribution.
    gen_w(self):
        Generate pulse widths (ms).
    generate_frbs(survey_model, beam_model, z_model, n_srcs, z_max,
                  bol_lum_low, specif_lum_high, w_min):
        Convenience function around frbpoppy, with a few changes.
    plot_population(frbs, cosmic_pop, cpop_factor, plot_james):
        Make a three-panel plot with different generated parameter distributions.

Created on Tue May  9 17:47:00 2023
"""
import types
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import lognorm

from frbpoppy.w_dists import calc_w_arr
from frbpoppy import CosmicPopulation, Survey, SurveyPopulation


def lognorm_mu(mu, sigma, w_min=0., shape=1, z=0):
    """Draw from a lognormal distribution.

    Use mu and sigma instead of mean and stddev, which frbpoppy uses.
    The function is used to generate intrinsic pulse widths.
    """
    draw_min = lognorm.cdf(w_min, s=sigma, loc=0, scale=mu)

    # Draw from uniform distribution and invert it. Note that scipys
    # lognorm is differently defined from numpy.randoms.
    rng = np.random.default_rng()
    uni = rng.uniform(low=draw_min, high=1., size=shape)
    w_int = lognorm.ppf(uni, s=sigma, loc=0, scale=mu).astype(np.float32)
    w_arr = calc_w_arr(w_int, z=z)
    return w_int, w_arr


def gen_w(self):
    """Generate pulse widths (ms).

    Modified to give redshift to function.
    """
    shape = self.w_shape()
    z = self.frbs.z
    self.frbs.w_int, self.frbs.w_arr = self.w_func(shape, z)

    # From combined distribution inputs
    if self._transpose_w:
        self.frbs.w_int = self.frbs.w_int.T
        self.frbs.w_arr = self.frbs.w_arr.T


def generate_frbs(survey_model, beam_model='gaussian', z_model='sfr', n_srcs=1e7, z_max=1.5,
                  bol_lum_low=1e40, specif_lum_high=1e40):
    """Generate a population of Fast Radio Bursts (FRBs).

    This function generates a population of FRBs using the frbpoppy
    module. It is mostly a convenience function to access the
    parameters we need. Most of the parameter values are
    based on the built-in "optimal" population in frbpoppy.
    All parameters are in some way forwarded to frbpoppy and are
    explained there in more detail.

    Args:
        survey_model (str): Name of the survey model to use.
            Choose from available survey names in frbpoppy.
        beam_model (str): Beam pattern to use for FRB detection.
            Choose from 'wsrt-apertif', 'parkes-htru', 'chime-frb',
            'gaussian', 'airy'.
        z_model (str): Cosmic number density model to use. Choose from
            'vol_co', 'sfr', 'smd'.
        n_srcs (float): Number of cosmic FRBs to generate.
        z_max (float): Maximum redshift of FRBs.
        bol_lum_low (float): Lower limit of the bolometric luminosity.
        specif_lum_high (float in erg/s/Hz): Upper limit, but in terms
            of the specific luminosity.

    Returns:
        Two frbpoppy populations: the cosmic population and the survey
        population.
    """
    spectral_index = -0.65
    # Generate an observed FRB population
    if isinstance(survey_model, str):
        survey = Survey(survey_model)
    else:
        survey = survey_model
    survey.set_beam(model=beam_model)
    ra_min, ra_max, dec_min, dec_max = survey.ra_min, survey.ra_max, survey.dec_min, survey.dec_max

    cosmic_pop = CosmicPopulation(n_srcs)
    cosmic_pop.set_dist(model=z_model, z_max=z_max)  # , alpha=-2.2)
    cosmic_pop.set_direction(min_ra=ra_min, max_ra=ra_max, min_dec=dec_min, max_dec=dec_max)
    cosmic_pop.set_dm(mw=False, igm=False, host=False)   # We don't care about DM

    # Emission frequency range.
    freq_low, freq_high = 100e6, 50e9
    cosmic_pop.set_emission_range(low=freq_low, high=freq_high)  # Default frequency

    # Luminosity function.
    freq_factor = (freq_high**(1+spectral_index) - freq_low**(1+spectral_index)
                   ) / 1.3e9**spectral_index
    luminosity_low = bol_lum_low  # bol_lum_low * freq_factor
    luminosity_high = specif_lum_high * freq_factor
    cosmic_pop.set_lum(model='powerlaw', low=luminosity_low, high=luminosity_high, power=-1.05)

    # Use our own function for the widths.
    cosmic_pop.w_shape = lambda: cosmic_pop.n_srcs
    cosmic_pop.w_func = lambda shape, z: lognorm_mu(mu=5.49, sigma=np.log(2.46), w_min=0.,
                                                    shape=shape, z=z)

    # To change the funtion in the Class we need this "types" from
    # stackoverflow. Would be good to put it in frbpoppy.
    # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
    cosmic_pop.gen_w = types.MethodType(gen_w, cosmic_pop)

    cosmic_pop.set_si(model='constant', value=spectral_index)
    cosmic_pop.generate()

    survey_pop = SurveyPopulation(cosmic_pop, survey)  # , scat=True, scin=True) These take long

    return cosmic_pop, survey_pop


# For comparison take the values from simulated FRBs of James et al. 2022
james_frbs = np.array([186.7, 15.9, 0.15, 898.1, 32.2, 0.718, 1179.5, 26, 1.321, 582.2, 13.8,
    0.571, 438.5, 20.2, 0.062, 636.2, 23.1, 0.441, 315.3, 17.9, 0.089, 735.7, 27.7, 0.482, 833.1,
    10.5, 0.949, 1405, 11.4, 1.318, 595.3, 32.8, 0.25, 1083, 19.3, 1.101, 313.4, 16.8, 0.37, 709.4,
    12.4, 0.58, 568.5, 13.1, 0.629, 1794.1, 14.9, 1.849, 143.5, 11, 0.026, 736.2, 11, 0.546, 743.4,
    12.6, 0.812, 808.6, 33.4, 0.472, 941.9, 10, 0.755, 352, 16, 0.126, 460.5, 10.7, 0.355, 447.5,
    38.3, 0.543, 1271.8, 12.1, 0.432, 1346.2, 12.3, 1.567, 1308.6, 15.7, 1.273, 428.3, 12.3, 0.537,
    567.9, 12.4, 0.608, 421.8, 28, 0.018, 410.5, 84.7, 0.057, 602.3, 16.3, 0.093, 391.1, 117.3,
    0.251, 1110.1, 94, 0.439, 1287.8, 9.7, 1.592, 303.5, 10.8, 0.137, 634.8, 15.9, 0.378, 799.7,
    18.8, 0.465, 383, 15.6, 0.176, 309.6, 30, 0.159, 372.7, 16.2, 0.167, 3446.6, 25.6, 0.08, 237.3,
    13.3, 0.041, 721.5, 25.8, 0.306, 478.8, 27, 0.354, 296.3, 21.4, 0.134, 835, 51.2, 0.735, 573.2,
    19.6, 0.464, 282.6, 10.6, 0.263, 184.5, 11.4, 0.063, 151.4, 223.4, 0.046, 580, 15.6, 0.632,
    819.9, 13.1, 0.75, 754, 9.6, 0.103, 162.3, 17.3, 0.138, 391.9, 11.2, 0.43, 371.1, 25, 0.282,
    282, 21.7, 0.073, 357.4, 16.3, 0.262, 548.7, 25.8, 0.35, 331.8, 40.2, 0.085, 449.7, 12.6,
    0.019, 557.5, 88.5, 0.289, 187.9, 40.8, 0.052, 818.5, 9.8,  0.66, 522.1, 11.8, 0.353, 1257.6,
    22.4, 0.699, 233.3, 85.5, 0.081, 1116.3, 11.6, 1.367, 923.9, 9.9, 0.961, 2259.2, 12.3, 1.338,
    568.1, 15.9, 0.295, 307.9, 22.6, 0.243, 1327.1, 18.6, 0.437, 1311.1, 24.4, 1.712, 901, 39.8,
    0.744, 848.2, 11.6, 1.006, 776.9, 14.5, 0.786, 1060.6, 10.6, 1.108, 359.8, 46.4, 0.372, 785.5,
    10, 0.164, 733.7, 18.7, 0.633, 484.9, 11.7, 0.52, 685.6, 9.8, 0.765, 481.2, 9.6, 0.424, 568.9,
    14.8, 0.644, 484.7, 14.9, 0.61, 398.7, 23.3, 0.239, 260.6, 12.9, 0.054, 664.2, 15.8, 0.495,
    393.8, 12.7, 0.291, 326.6, 14.2, 0.251, 273.7, 22.7, 0.182, 726, 23.7, 0.089, 534.4, 21, 0.556,
    342.1, 48.2, 0.029, 703.6, 10.8, 0.195, 376.5, 47.9, 0.279, 335.9, 18.5, 0.329, 271.9, 18.9,
    0.237, ])


def plot_population(frbs, cosmic_pop, cpop_factor=1, plot_james=True):
    """Make a three panel plot with different generated parameter distributions."""
    # Define some plotting variables.
    textwidth = 7.0282  # might also be from latex textwidth=17.85162cm columnwidth=8.5744cm
    small_size = 8
    medium_size = 10
    bigger_size = 12

    sns.set_theme(style="ticks", context="paper")

    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,
                                        figsize=(textwidth, textwidth/3*4.8/6.4),
                                        gridspec_kw={'wspace': 0.06})

    palette = sns.color_palette('Set2')

    wbins = np.logspace(-1, 3., 30)
    histo, bins = np.histogram(cosmic_pop.frbs.w_int, bins=wbins)
    ax1.bar(bins[:-1], cpop_factor*histo, width=np.diff(bins), align='edge',
            label='cosmic population', color=palette[2])
    ax1.hist(frbs['w_int'], density=False, bins=wbins, alpha=1., color=palette[1],
             label='observed population')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$w_\mathrm{int}$ (ms)')
    ax1.set_ylabel('Number of bursts')
    ax1.set_xticks([.1, 1, 10, 100, 1e3])

    # Luminosity
    l_min, l_max = cosmic_pop.frbs.lum_bol.min(), cosmic_pop.frbs.lum_bol.max()
    lbins = np.logspace(np.log10(l_min), np.log10(l_max), 30)
    histo, bins = np.histogram(cosmic_pop.frbs.lum_bol, bins=lbins)

    ax2.bar(bins[:-1], cpop_factor*histo, width=np.diff(bins), align='edge', color=palette[2],
            label='cosmic population')
    ax2.hist(frbs['lum_bol'], density=False, bins=lbins, alpha=1, color=palette[1],
             label='observed population')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$L_\mathrm{bol} \;(\mathrm{erg\,s}^{-1})$')
    #ax2.set_xticks([1e40, 1e42, 1e44])
    ax2.legend()

    # Redshift
    histo, bins = np.histogram(cosmic_pop.frbs.z, bins=30, range=(0, cosmic_pop.frbs.z.max()))

    ax3.bar(bins[:-1], cpop_factor*histo, align='edge', width=np.diff(bins), color=palette[2])
    ax3.hist(frbs['z'], density=False, bins=bins, alpha=1., color=palette[1])

    if plot_james:
        # Add simulated FRBs from James et al. 2022
        james_zs = james_frbs.reshape((100, 3))[:, 2]
        # Make the same binsize as other data.
        if bins[-1] < james_zs.max():
            nbins = np.ceil(james_zs.max()/(bins[1]-bins[0]))
            bins = np.linspace(0, nbins*(bins[1]-bins[0]), num=int(nbins))
        ax3.hist(james_zs, density=False, bins=bins, alpha=.8, color=palette[0],
                 label="James et al. (2022c)")
        ax3.legend()

    ax3.set_yscale('log')
    ax3.set_xlabel('$z$')
    ax3.set_ylim(0.5, 1e10)
    ax3.set_yticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])

    return fig
