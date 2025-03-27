#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core module to assess observability of FRB hosts.

This module provides various functions to draw host galaxies, assess
visibility in optical surveys, and calculate required observing times.

Functions:
    get_file_zs(gal_file, header_row): Retrieve redshift of a GALFORM
        file.
    get_file_galaxies(gal_file, header_row): Extract galaxies from a
        GALFORM file.
    download_galaxies(local_dir): Download GALFORM data to a local
        directory.
    draw_galaxies(frb_zs, weights, seed, gal_files): Draw galaxies based
        on given redshifts.
    observed_bands(frbs, galaxies): Observe FRB hosts with different
        telescopes.
    observing_time(mag, snr, gal_size): Calculate the necessary
        observing time for a photometric band.
    observing_time_spectrum(mag, snr, gal_diam): Calculate the necessary
        observing time for a spectrometer.
    draw_DM(frb_zs, F, mu, lognorm_s, rng): Draw DM values for FRBs
        based on their redshifts.
    draw_Delta(z, f, n_samples, rng): Draw Delta values given a
        redshift.
    average_DM_deviation(c0, sigma): Calculate the deviation of the
        average DM from its expected value for a given C_0.

Created on Tue Mar  1 13:09:40 2022
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mockFRBhosts

from glob import glob
from urllib.request import urlopen, urlretrieve
from numpy.random import normal
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.special import hyp1f1
from scipy.special import gamma
from scipy.stats import sigmaclip
from scipy.stats import lognorm
from astropy.coordinates import Distance
from astropy.cosmology import Planck18 as cosmo

# Silence warnings about the hmf module.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from frb.dm.cosmic import DMcosmic_PDF
    from frb.dm.igm import average_DM


def get_file_zs(gal_file, header_row=4):
    """Get the redshift from the file header."""
    with open(gal_file, 'r') as f:
        header = ''
        for _ in range(header_row):
            header += f.readline()

    # Find the pattern and take the first thing afterwards.
    z_pattern = 'redshift z='
    redshift = header[header.rfind(z_pattern) + len(z_pattern):]
    redshift = redshift.split()[0]
    return float(redshift)


def get_file_galaxies(gal_file, header_row=4):
    """Read the data excluding the # at the header line."""
    # Raise error if file does not exist. Otherwise pandas just crashes the kernel.
    if not os.path.isfile(gal_file):
        raise FileNotFoundError(f"{gal_file} does not exist.")

    headers = pd.read_csv(gal_file, delim_whitespace=True, header=header_row, nrows=0).columns[1:]
    data = pd.read_csv(gal_file, names=headers, delim_whitespace=True, header=None,
                       skiprows=header_row+1, skipfooter=1, engine='python')
    return data


def download_galaxies(local_dir):
    """Download GALFORM galaxies from Zenodo."""
    # Find all files on the page
    url = 'https://zenodo.org/record/7926078'
    page = urlopen(url)
    html = page.read().decode("utf-8")
    files = re.findall(r'https://zenodo.org/record.*?frb\.cat', html)

    # Ensure directory exists.
    if not os.path.isdir(local_dir):
        os.mkdir(local_dir)
    print("Galaxy data is being downloaded, this may take a few minutes (784 MB).")
    for file in files:
        # Define the local filename to save data
        local_file = os.path.join(local_dir, os.path.basename(file))
        # Download remote and save locally
        urlretrieve(file, local_file)


def draw_galaxies(frb_zs, weights='mstardot', seed=None, gal_files=None):
    """Draw galaxies from a database for given redshifts.

    Args:
        frb_zs (numpy.ndarray): Redshifts for which galaxies are to be
            drawn.
        weights (str, optional): Weighting scheme for galaxy selection.
            Choices are 'mstardot' or 'mstars_total'.
        seed (int, optional): Seed for the random number generator.
        gal_files (list, optional): List of galaxy files. If not
            provided, GALFORM galaxy files will be automatically
            searched and downloaded.

    Returns:
        pandas.DataFrame: Dataframe of drawn galaxies.
        numpy.ndarray: Snapshots number in which the FRBs fell.
    """
    if not gal_files:
        # Find files with galaxies from GALFORM files.
        module_path = os.path.dirname(mockFRBhosts.__file__)
        gal_path = os.path.join(module_path.rsplit('/', 1)[0], 'GALFORM_galaxies')
        gal_files = glob(gal_path + '/**/*frb.cat', recursive=True)
        if not gal_files:
            download_galaxies(gal_path)
            gal_files = glob(gal_path + '/**/*frb.cat', recursive=True)

    # Get the redshifts.
    snap_zs = np.array([get_file_zs(gal_file) for gal_file in gal_files])

    # Sort by redshift.
    z_order = snap_zs.argsort()
    snap_zs = snap_zs[z_order]
    if np.any(snap_zs[:-1] == snap_zs[1:]):
        raise ValueError("There are duplicats in the snapshots, it's likely iz105 which is weird.")
    gal_files = [gal_files[i] for i in z_order]

    # Make bins around the catalog snapshots
    bins = np.concatenate((snap_zs[0:1], snap_zs[:-1] + np.diff(snap_zs)/2, snap_zs[-1:] + 50.))
    nums_to_draw = np.histogram(frb_zs, bins=bins)[0]
    relev_snap = np.nonzero(nums_to_draw != 0)[0]

    # Draw galaxies from the GALFORM files. This assumes they are sorted like their redshifts.
    rng = np.random.default_rng(seed)
    galaxies = []
    for snap, n_draw in zip(relev_snap, nums_to_draw[relev_snap]):
        galaxy_pop = get_file_galaxies(gal_files[snap])
        galaxy_pop.loc[:, 'redshift'] = snap_zs[snap]
        galaxy_pop.loc[:, 'snapnum'] = snap
        galaxies.append(galaxy_pop.sample(n=n_draw, replace=False, weights=weights,
                                          random_state=rng))

    galaxies = pd.concat(galaxies)
    # Order FRBs such that they correspond to galaxies at the same positions.
    snapnum = np.digitize(frb_zs.to_numpy(), bins) - 1

    return galaxies, snapnum


def observed_bands(frbs, galaxies):
    """Observe FRB hosts with different telescopes.

    Args:
        frbs (numpy.ndarray or pandas.DataFrame): Array or DataFrame
            containing FRB data.
        galaxies (pandas.DataFrame): DataFrame containing galaxy data.

    Returns:
        numpy.ndarray: Number of observable bands in SDSS.
        numpy.ndarray: Number of observable bands in LSST.
        numpy.ndarray: Number of observable bands in EUCLID.
        numpy.ndarray: Number of observable bands in DES.
    """

    if not np.all(frbs['snapnum'].to_numpy() == galaxies['snapnum'].to_numpy()):
        warnings.warn("FRBs and galaxies are not ordered in the same manner.")

    if not isinstance(frbs, np.ndarray):
        frb_zs = frbs['z'].to_numpy()
    else:
        frb_zs = frbs

    dist = Distance(z=frb_zs, cosmology=cosmo)

    # Test if the galaxies would be observable in the LSST.
    # Limits taken from https://www.lsst.org/scientists/keynumbers
    # 5-sigma point source depth: Single exposure and idealized for
    # stationary sources after 10 years,
    # u : 23.9,  26.1
    # g : 25.0,  27.4
    # r : 24.7, 27.5
    # i : 24.0 , 26.8
    # z : 23.3,  26.1
    # y : 22.1, 24.9
    apparent_mag = (dist.distmod.value[:, np.newaxis] + 5*np.log10(cosmo.h)
                    - 2.5*np.log10(1+frb_zs)[:, np.newaxis]
                    + galaxies.loc[:, 'mag_LSST-u_tot':'mag_LSST-y_tot'])
    # Single exposure only gets galaxies z<~0.6
    # mag_limits_LSST = np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1])
    mag_limits_LSST = np.array([26.1, 27.4, 27.5, 26.8, 26.1, 24.9])
    n_bands_obs_LSST = (apparent_mag < mag_limits_LSST).sum(axis=1)

    # Test if the galaxies would be observable in EUCLID.
    # Limits taken from https://sci.esa.int/web/euclid/-/euclid-vis-instrument
    # updated from https://ui.adsabs.harvard.edu/abs/2022A%26A...662A.112E/abstract
    apparent_mag = (dist.distmod.value[:, np.newaxis] + 5*np.log10(cosmo.h)
                    - 2.5*np.log10(1+frb_zs)[:, np.newaxis]
                    + galaxies.loc[:, 'mag_EUCLID-Y_t':'mag_EUCLID-K_t'])
    mag_limits_Euclid = np.array([26.2, 24.5, 24.5, 24.5])
    n_bands_obs_Euclid = (apparent_mag < mag_limits_Euclid).sum(axis=1)

    # Test observability in DES
    apparent_mag = (dist.distmod.value[:, np.newaxis] + 5*np.log10(cosmo.h)
                    - 2.5*np.log10(1+frb_zs)[:, np.newaxis]
                    + galaxies.loc[:, 'mag_DES-g_tot_':'mag_DES-z_tot_'])
    mag_limits_DES = np.array([24.3, 23.9, 23.5, 22.8])
    n_bands_obs_DES = (apparent_mag < mag_limits_DES).sum(axis=1)

    # Test if the galaxies would be observable in SDSS DR16.
    # Limits taken from https://www.sdss4.org/dr17/scope/
    apparent_mag = (dist.distmod.value[:, np.newaxis] + 5*np.log10(cosmo.h)
                    - 2.5*np.log10(1+frb_zs)[:, np.newaxis]
                    + galaxies.loc[:, 'mag_SDSS-u_tot':'mag_SDSS-z_tot'])
    mag_limits_SDSS = np.array([22.0, 22.2, 22.2, 21.3, 20.5])
    n_bands_obs_SDSS = (apparent_mag < mag_limits_SDSS).sum(axis=1)

    return n_bands_obs_SDSS, n_bands_obs_LSST, n_bands_obs_Euclid, n_bands_obs_DES


def observing_time(mag, snr=10, gal_size=1):
    """Calculate the necessary observing time for an example telescope.

    This function follows Chapter 17 of Schroeder (2000) "Astronomical
    Optics". It calculates the observing time in the background-limited
    case for a fixed set of telescope parameters.

    Args:
        mag (float or array-like): Apparent magnitudes.
        snr (float): Desired signal-to-noise ratio.
        gal_size (float or array-like): Galaxy size on the sky
            (arcsec^2).

    Returns:
        array-like: Observing time (s).
    """
    kappa = 0.8  # instrumental losses, not included in tau
    Q = 0.8
    tau = 0.3  # system transmittance
    diameter = 1000  # telescope diameter (cm)
    area = 0.7*diameter**2  # effective collecting area
    N0 = 1e4  # magnitude to flux conversion factor
    # (photons/(s cm^2 nm)) equivalent to around 3644 Jy at 550 nm
    bandpass = 100  # bandpass (nm)
    m_b = 22  # background brightness (mag/arcsec**2)

    # Calculate the counts in the detector from source and the background
    S = tau*area*N0*bandpass*10**(-0.4*mag)
    B = tau*area*N0*bandpass*10**(-0.4*m_b)*gal_size
    return B/Q*(snr/(kappa*S))**2


def observing_time_spectrum(mag, snr=10, gal_diam=2):
    """Calculate the required observing time for a spectrometer.

    This function follows chapter 17 of Schroeder (2000) "Astronomical
    Optics". We calculate the observing time in the background-limited
    case for a fixed set of telescope parameters. The difference from
    the observing_time function is that (at least) one direction on the
    sky is limited by the slit width, and that the band pass is instead
    the line width. We assume that the the galaxy size is 1arcsec^2.

    Args:
        mag (float or array): Apparent magnitudes.
        snr (float): Desired signal-to-noise ratio.
        gal_diam (float or array): Angular size of the galaxy along the
            long side of the slit (arcsec).

    Returns:
        Array: Observing time (s)
    """
    kappa = 0.8  # instrumental losses, not included in tau
    Q = 0.8
    tau = 0.3  # system transmittance
    diameter = 1000  # telescope diameter (cm)
    area = 0.7*diameter**2  # effective collecting area
    N0 = 1e4  # magnitude to flux conversion factor
    # (photons/(s cm^2 nm)) equivalent to around 3644 Jy at 550 nm
    bandpass = 1  # bandpass (nm)
    m_b = 22  # background brightness (mag/arcsec**2)
    wp = 0.003  # w': reimaged slit at camera focus (cm) (equivalent to 30um)
    r = 0.9  # anamorphic magnification
    F_2 = 1.5  # camera optics focal length / diameter of collimated
    # beam, incident on the disperser (p. 308)
    phi1 = wp/(r*diameter*F_2)*180*3600/np.pi  # convert to arcsec
    # Consider the case that the galaxy is completely covered by the slit.
    covered = np.where(phi1 < gal_diam, phi1/gal_diam, 1)

    # Calculate the counts in the detector from source and the background.
    S = tau*area*N0*bandpass*10**(-0.4*mag)*covered
    B = tau*area*N0*bandpass*10**(-0.4*m_b)*phi1*gal_diam

    return B/Q*(snr/(kappa*S))**2


def draw_DM(frb_zs, F=0.2, mu=100, lognorm_s=1, rng=None):
    """Draw a DM for each given reshift.

    Given the parameter values from this function simulates the DM.
    Obh70 can not be given at the moment, would have to give it to
    averag_DM.

    Args:
        frb_zs (array-like): Redshifts of FRBs.
        F (float): Fraction of baryonic DM in the intergalactic medium.
        mu (float): Mean of the lognormal distribution for DM_host.
        lognorm_s (float): Standard deviation of the lognormal
            distribution for DM_host.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        numpy.ndarray: Total dispersion measures (DM) for the FRBs.

    """
    # Calculate the average DM up to the highest redshift, interpolate
    # to avoid using this slow function again. (For every neval an
    # integral is done in frb.dm.igm.avg_rhoISM when cosmo.age(z) is called.)
    DM_cum, zeval = average_DM(frb_zs.max(), cosmo=cosmo, cumul=True)
    avrg_DM = interp1d(zeval, DM_cum, assume_sorted=True)

    # Draw a Delta from it's PDF. Multiply by <DM_cosmic> to get a DM.
    delta = [float(draw_Delta(z, f=F, n_samples=1, rng=rng)) for z in frb_zs]
    dm_cosmic = np.array(delta) * avrg_DM(frb_zs)

    # Draw a DM_host.
    dm_host = lognorm.rvs(lognorm_s, scale=mu, size=len(frb_zs))

    return dm_host/(1+frb_zs) + dm_cosmic


def draw_Delta(z, f=0.2, n_samples=1, rng=None):
    """Draw Delta from p_cosmic.

    Following Macquart et al. 2020 the PDF can be described by their
    equation (4). Here Delta = DM_cosmic / <DM_cosmic>, that means to
    get DM_cosmic the returned number has to be multiplied by the
    average of DM_cosmic. This is because frb.dm.igm.average_DM() is
    very slow and should only be used once (with cumul=True) and then be
    interpolated.

    Args:
        z (float): Redshift.
        f (float): Strength of baryon feedback F.
        n_samples (int): Number to draw.
        rng (numpy.random.Generator, optional): Random number generator.

    Returns:
        numpy.ndarray: Delta for given z, defined as
            DM_cosmic / <DM_cosmic>.
    """
    sigma = f/np.sqrt(z)
    c0 = minimize_scalar(average_DM_deviation, args=(sigma)).x

    hyp_x = -c0**2/18/sigma**2
    normalization = 3*(12*sigma)**(1/3)/(gamma(1/3)*3*sigma*hyp1f1(1/6, 1/2, hyp_x)
                                         + 2**(1/2)*c0*gamma(5/6)*hyp1f1(2/3, 3/2, hyp_x))

    # Create 20000 values of the PDF to create the inverse from.
    Delta_values = np.linspace(1/1000., 20., 20000)  # error at z=10 is <0.005%
    # dpm3 = Delta_values**-3  # Delta to the power of -3
    # pdf = normalization * np.exp(-(dpm3 - c0) ** 2 / (18 * sigma*sigma)) * dpm3

    pdf = DMcosmic_PDF(Delta_values, c0, sigma, A=normalization, alpha=3., beta=3.)

    # Invert the CDF.
    cum_values = pdf.cumsum()/pdf.sum()
    inv_cdf = interp1d(cum_values, Delta_values, assume_sorted=True)

    if rng is None:
        rng = np.random.default_rng()
    r = rng.random(n_samples)

    return inv_cdf(r)


def average_DM_deviation(c0, sigma):
    """Take the difference between the average Delta and its target.

    C_0 is not a free parameter, but is fixed by the definition
    Delta = DM_cosmic / <DM_cosmic> to guarantee <Delta> = 1.
    """
    # Compute x that is used in every hyp1f1
    hyp_x = -c0**2/18/sigma**2

    normalization = 3*(12*sigma)**(1/3)/(gamma(1/3)*3*sigma*hyp1f1(1/6, 1/2, hyp_x)
                                         + 2**(1/2)*c0*gamma(5/6)*hyp1f1(2/3, 3/2, hyp_x))
    # normalization = 3*np.cbrt(12*sigma)/(gamma(1/3)*3*sigma*hyp1f1(1/6, 1/2, hyp_x)
    #                                     + np.sqrt(2)*c0*gamma(5/6)*hyp1f1(2/3, 3/2, hyp_x))

    avrg_DM = normalization/3 * (gamma(1/6)*hyp1f1(1/3, 1/2, hyp_x)/(2**5/9/sigma**2)**(1/6)
                                 + c0*gamma(2/3)*hyp1f1(5/6, 3/2, hyp_x)/(18*sigma**2)**(1/3))
    return np.abs(avrg_DM-1)


# From Lauras UsefulNotebooks/SDSS_MagErr.ipynb
def estimate_photo_err(mags, real_mags, real_errs, bins=30, doplot=False, mag2plot=20):
    """Estimate errors from the data itself.

    From a set of input photometric magnitudes, it will estimate the errors assuming
    lognormal statistics.
    The median and clipped std dev are used to determine lognormal properties.
    One can choose to plot the real errors and estimated errors together.

    Input:
    myband = one of the five SDSS photometric bands
    bins = an array that defines the binning
    doplot = if True, plot the true errors from "myband" and estimated errors

    Returns:
       Array of estimated errors for each input magnitude
       Array of measured median and clipped std. dev. for each mag. bin
    """
    # Create bins.
    if isinstance(bins, int):
        bins = np.linspace(real_mags.min(), real_mags.max() + 1e-9, bins)

    # Make sure bins sufficient real_mags to determine statistics.
    empty_bins = True
    while empty_bins:
        # Bin the magnitudes. The statistics will be calculated in each bin.
        bin_nr = np.digitize(real_mags, bins)

        # See which bins are empty
        nr_in_bins = np.histogram(bin_nr, bins=np.arange(0.5, len(bins), 1.))[0]
        empty = np.nonzero(nr_in_bins <= 1)[0]  # also if only one

        if empty.size > 0:
            # Remove the empty bin, make a new bin edge in the middle.
            bins[empty] = (bins[empty]+bins[empty+1])/2
            bins = np.delete(bins, empty+1)
        else:
            empty_bins = False

    # Do the same with the simulated magnitudes.
    new_err_bin = np.digitize(mags, bins)
    # Place magnitudes beyond the bins in the outermost bins.
    new_err_bin[new_err_bin == 0] = 1
    new_err_bin[new_err_bin == len(bins)] = len(bins) - 1

    nbins = len(bins)
    # Initialize some arrays
    err_med = np.zeros(nbins)
    err_std = np.zeros(nbins)
    myerr = np.full(len(mags), np.nan)

    # Loop over bins
    for i in range(1, nbins):
        # Pull out galaxies that are in the current bin of interest
        log_real_errs = np.log10(real_errs[bin_nr == i])
        bin_mags = new_err_bin == i

        # Median
        err_med[i] = np.median(log_real_errs)

        # Clipped stddev (this removes outliers)
        cliped_errs, _, _ = sigmaclip(log_real_errs)
        err_std[i] = np.std(cliped_errs)

        # Simulate Gaussian noise
        mynoise = normal(err_med[i], err_std[i], size=(bin_mags.sum(),))

        # Put these back into the full array
        myerr[bin_mags] = 10**mynoise

    if doplot:
        imag = np.where(bins >= mag2plot)[0][0]
        mymag = np.where(bin_nr == imag)[0]
        lmag = len(mymag)

        plt.figure()
        plt.hist(real_errs[mymag], bins=100)
        plt.hist(10**normal(err_med, err_std[i], size=(lmag,)), histtype='step',
                 label='Med = %f\nStd = %f' % (err_med[imag], err_std[imag]), bins=100)
        plt.legend()
        plt.title("m = %f" % mag2plot)

    leftover_nans = np.isnan(myerr).sum()
    if leftover_nans:
        print(leftover_nans, "magnitudes did not get an uncertainty")

    return myerr, err_med, err_std, bins


def beck_mag_cuts(bands, errs, coloronly=False, verbose=False):
    """
    Input:
    uband, gband, rband, iband, zband = 2D numpy arrays containing the
    (ugriz)-band magnitudes and their errors
        e.g. uband[0] = array of magnitudes
             uband[1] = array of magnitude errors
    coloronly = Make cuts based only on colour and not not errors (useful for debugging)
    verbose = if True, print how many galaxies remain after each cut

    Returns:
        An array containing the indexes of galaxies that made the cut
    """
    # Calculate color differences. Could use bands.diff(periods=-1, axis=1)
    ugcolor = bands['u'] - bands['g']
    grcolor = bands['g'] - bands['r']
    ricolor = bands['r'] - bands['i']
    izcolor = bands['i'] - bands['z']

    # Estimate color errors.
    # This assumes no covariances in errors between different bands
    err_gr = np.sqrt(errs['err_g']**2 + errs['err_r']**2)
    err_ri = np.sqrt(errs['err_r']**2 + errs['err_i']**2)
    err_iz = np.sqrt(errs['err_i']**2 + errs['err_z']**2)

    # Cuts on color
    ug_cond = (ugcolor > -0.911) & (ugcolor < 5.597)
    gr_cond = (grcolor > 0.167) & (grcolor < 2.483)
    ri_cond = (ricolor > 0.029) & (ricolor < 1.369)
    iz_cond = (izcolor > -0.542) & (izcolor < 0.790)

    # Cuts on magnitude error
    dr_cond = errs['err_r'] < 0.15
    dgr_cond = err_gr < 0.225
    dri_cond = err_ri < 0.15
    diz_cond = err_iz < 0.25

    # Take the intersection of all conditions
    condition = ug_cond & gr_cond & ri_cond & iz_cond

    if not coloronly:
        error_cond = dr_cond & dgr_cond & dri_cond & diz_cond
        condition &= error_cond

    if verbose:
        print("Original length: ", len(bands))
        print("Valid (u-g): ", np.count_nonzero(ug_cond))
        print("Valid (g-r): ", np.count_nonzero(gr_cond))
        print("Valid (r-i): ", np.count_nonzero(ri_cond))
        print("Valid (i-z): ", np.count_nonzero(iz_cond))

        if not coloronly:
            print("Valid d(r): ", np.count_nonzero(dr_cond))
            print("Valid d(g-r): ", np.count_nonzero(dgr_cond))
            print("Valid d(r-i): ", np.count_nonzero(dri_cond))
            print("Valid d(i-z): ", np.count_nonzero(diz_cond))

        print("All valid: ", np.count_nonzero(condition))

    return condition
