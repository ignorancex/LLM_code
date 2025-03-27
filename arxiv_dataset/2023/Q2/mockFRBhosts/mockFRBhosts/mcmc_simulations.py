#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides functions to infer parameters of FRBs.

To infer the cosmological parameters F and O_m*H_0 and the FRB host DM
properties, this module provides the functions to do a MCMC simulations.
The method is based on, and uses the code from Macquart et al. 2020.
It includes Likelihood functions where an error in the normalization
has been corrected w.r.t. the frb.dm package.

Functions:
    do_mcmc(frb_zs, frb_DMs, draws, cores, tune): Do the MCMC
        simulation.
    log_likelihood(Obh70, F, in_DM_FRBp, z_FRB, mu, lognorm_s,
                   lognorm_floor, beta, step): Likelihood that
        marginalizes by integrating with a fixed step size. Currently
        not used.
    log_likelihood_variable_step(Obh70, F, in_DM_FRBp, z_FRB, mu,
                                 lognorm_s, beta, res): Likelihood that
        marginalizes by integrating with a given resolution.

Created on Sat Jul  9 01:16:51 2022
"""
import warnings
import numpy as np
import pymc3 as pm

from scipy.stats import lognorm
from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from astropy.cosmology import Planck18 as cosmo

# Silence warnings about the hmf module.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from frb import defs
    from frb.dm import mcmc
    from frb.dm import cosmic
    from frb.dm.igm import average_DM


def do_mcmc(frb_zs, frb_DMs, draws, cores=4, tune=300):
    """MCMC to reproduce the last figure of Macquart et al. 2020.

    Adapted from frb.tests.test_mcmc.test_pm

    Args:
        frb_zs (pandas.DataFrame): FRB redshifts.
        frb_DMs (pandas.DataFrame): FRB dispersion measures.
        draws (int): MCMC draws done after the tune phase per chain.
        cores (int): Number of CPU cores to use. Each core simulates
            its own MCMC chain.
        tune (int): Number of tuning draws.

    Returns:
        arviz.InferenceData: Draws of all chains.
    """
    # Calculate the average DM up to the highest redshift, interpolate to avoid using this slow
    # function again.
    # (For every neval an integral is done in frb.dm.igm.avg_rhoISM when cosmo.age(z) is called.)
    DM_cum, zeval = average_DM(frb_zs.max(), cosmo=defs.frb_cosmo, cumul=True)

    global f_C0_3, spl_DMc, cosmo_Obh70
    f_C0_3 = cosmic.grab_C0_spline()
    spl_DMc = IUS(zeval, DM_cum.value)

    cosmo_Obh70 = cosmo.Ob0 * (cosmo.H0.value/70.)

    mcmc.all_prob = log_likelihood_variable_step
    mcmc.frb_zs = frb_zs.to_numpy()
    mcmc.frb_DMs = frb_DMs.to_numpy()

    parm_dict = mcmc.grab_parmdict()
    with mcmc.pm_four_parameter_model(parm_dict, beta=3.):
        idata = pm.sample(draws, tune=tune, cores=cores, return_inferencedata=True,
                          discard_tuned_samples=True, progressbar=False,)
    return idata


def log_likelihood(Obh70, F, in_DM_FRBp, z_FRB, mu=100.,
                   lognorm_s=1., lognorm_floor=0., beta=3., step=1.):
    """Calculate the probability for a set of FRBs.

    Compared to the previously used "all_prob", this function includes a
    factor of <DM_cosmic> that comes from changeing the integration to
    Delta, i.e. from dDM = dDelta * <DM_cosmic>. On top, this function
    has a significant speed up due to looping instead of leaving parts
    of an array empty. Using a variable step size instead did not seem
    as roburst.

    Args:
        Obh70 (float): Value of Omega_b * h_70
        F (float): Feedback parameter
        in_DM_FRBp (np.ndarray): Measured DMs with DM_MW already
            subtracted.
        z_FRB (np.ndarray): FRB redshifts.
        mu (float, optional):
            Mean of log-normal PDF of DM_host (in DM units).
        lognorm_s (float, optional):
            Sigma of log-normal PDF of DM_host (in log space).
        beta (float, optional): Parameter for DM PDF.
        step (float, optional):
            Step size in DM units for the integral over the DM.

    Returns:
        float:  Log likelihood
    """
    # Sigma
    sigma = F / np.sqrt(z_FRB)
    # C0
    if beta == 4.:
        raise ValueError("Bad beta value in all_prob")
    elif beta == 3.:
        C0 = f_C0_3(sigma)
    else:
        raise IOError

    likelihoods = np.zeros(z_FRB.shape)

    avgDM = spl_DMc(z_FRB) * (Obh70 / cosmo_Obh70)

    for i, in_DM in enumerate(in_DM_FRBp):
        DM_values = np.arange(step/2, in_DM, step)
        DMcosmic = in_DM - DM_values
        Delta = DMcosmic / avgDM[i]
        PDF_Cosmic = cosmic.DMcosmic_PDF(Delta, C0[i], sigma[i], beta=beta)
        PDF_host = lognorm.pdf(DM_values*(1+z_FRB[i]), lognorm_s, lognorm_floor, mu)
        likelihoods[i] = step*np.sum(PDF_Cosmic * PDF_host)

    if beta == 3.:
        # Calculate the normalization "analytically" to be fast.
        hyp_x = -C0**2/18/sigma**2
        normalizations = 3*(12*sigma)**(1/3)/(gamma(1/3)*3*sigma*hyp1f1(1/6, 1/2, hyp_x)
                                              + 2**(1/2)*C0*gamma(5/6)*hyp1f1(2/3, 3/2, hyp_x))
    else:
        # Integrate numerically. This is slower by a factor 146 (with 20000 samples).
        step = 20/20000
        Delta = np.linspace(step, 20.-step, 20000)
        normalizations = cosmic.DMcosmic_PDF(Delta, C0[:, np.newaxis], sigma[:, np.newaxis],
                                             beta=beta)
        normalizations = 1 / (step * normalizations.sum(axis=-1))

    log_like = np.sum(np.log(likelihoods*normalizations/avgDM))
    return log_like


def log_likelihood_variable_step(Obh70, F, in_DM_FRBp, z_FRB,
                                 mu=100., lognorm_s=1., lognorm_floor=0., beta=3., res=400):
    """Calculate the log likelihood for a set of FRBs.

    Compared to the previously used "all_prob", this function includes a
    factor of <DM_cosmic> that comes from changeing the integration to
    Delta, i.e. from dDM = dDelta * <DM_cosmic>. On top, this function
    has a significant speed up due to a variable stepsize and the use of
    broadcasting.

    Args:
        Obh70 (float): Value of Omega_b * h_70
        F (float): Feedback parameter
        in_DM_FRBp (np.ndarray): Measured DMs with DM_MW already
            subtracted.
        z_FRB (np.ndarray): FRB redshifts.
        mu (float, optional):
            Mean of log-normal PDF of DM_host (in DM units).
        lognorm_s (float, optional):
            Sigma of log-normal PDF of DM_host (in log space).
        lognorm_floor (arbitrary): It is not used.
        beta (float, optional): Parameter for DM PDF.
        res (int, optional):
            Number of steps to use for the integral over the DM.
            In a test with 200 FRBs the difference between res=100 and
            res=10000 was only 0.3%.

    Returns:
        float:  Log likelihood
    """
    # Sigma for each FRB.
    sigma = F / np.sqrt(z_FRB)
    # C0 for each FRB.
    if beta == 4.:
        raise ValueError("Bad beta value in all_prob")
    elif beta == 3.:
        C0 = f_C0_3(sigma)
    else:
        raise IOError

    # Get the average DM for each z from the globally created spline.
    avgDM = spl_DMc(z_FRB) * (Obh70 / cosmo_Obh70)

    # Integrate over P_host and P_cosmic in eq. 7 of Macquart et al. 2020 using the rectangle rule.
    steps = in_DM_FRBp/(res+1)  # Integration steps
    DM_values = np.linspace(steps/2, in_DM_FRBp-steps/2, res, axis=-1)  # 0th axis are the FRBs.

    Delta = (in_DM_FRBp[:, np.newaxis] - DM_values) / avgDM[:, np.newaxis]
    PDF_Cosmic = cosmic.DMcosmic_PDF(Delta, C0[:, np.newaxis], sigma[:, np.newaxis], beta=beta)
    PDF_host = lognorm.pdf(DM_values*(1+z_FRB[:, np.newaxis]), s=lognorm_s, scale=mu)
    likelihoods = steps*np.sum(PDF_Cosmic * PDF_host, axis=-1)

    if beta == 3.:
        # Calculate the normalization "analytically" to be fast.
        hyp_x = -C0**2/18/sigma**2
        normalizations = 3*(12*sigma)**(1/3)/(gamma(1/3)*3*sigma*hyp1f1(1/6, 1/2, hyp_x)
                                              + 2**(1/2)*C0*gamma(5/6)*hyp1f1(2/3, 3/2, hyp_x))
    else:
        # Integrate numerically. This is slower by a factor 146 (with 20000 samples).
        step = 20/20000
        Delta = np.linspace(step, 20.-step, 20000)
        normalizations = cosmic.DMcosmic_PDF(Delta, C0[:, np.newaxis], sigma[:, np.newaxis],
                                             beta=beta)
        normalizations = 1 / (step * normalizations.sum(axis=-1))

    # Normalization matters because it is different for each FRB. The factor avgDM comes because
    # normalizations is the integral over Delta instead of DM. avgDM was missing in previous
    # versions.
    log_like = np.sum(np.log(likelihoods*normalizations/avgDM))
    return log_like
