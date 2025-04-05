 #!/usr/bin/env python


__version__ = '0.1.4'

__revision__ = '20210128'

import sys
import os
import shutil
import warnings
import pyspeckit
import numpy as np
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from astropy import units as u
from astropy import constants as const
from astropy import log
from astropy.stats import median_absolute_deviation as MAD
import pandas as pd
import time
import logging
from pysynphot import observation
from pysynphot import spectrum
from pysynphot import ObsBandpass
from pysynphot import BlackBody
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
from scipy.special import wofz
from itertools import combinations as inter_comb


def initial_guesses(self, lines, blends=None, linestrength=100.,\
                    llimits=[2. * (-1), 2.]):

    '''
    Creates the initial guesses for the line fitter

    Args:
        lines : :func:`numpy.array`
            central wavelengths of the spectral lines

    Kwargs:
        linestrength : :obj:`float` (default: 100)
            initial guess for the line strength

        blends : :obj:`str` (optional)
            A file containing the a list of blended lines in the
            format: ** List is coming soon**

    return:
        guesses : :obj:`list`
            lists containing the guesses

        limits : :obj:`list`
            lists containing the limits

        limited : :obj:`list`
            lists containing the limited
    '''

    if self.linetype == 'absorption':
        linestrength = linestrength * (-1)

    guesses = np.zeros(4 * len(lines))
    limits = []
    limited = []

    for idx, line in enumerate(lines):

        if self.rv_sys != 0.:
            line = lambda_rv_shift(self, line)

        guesses[4 * idx + 0] = linestrength
        guesses[4 * idx + 1] = line
        guesses[4 * idx + 2] = self.dispersion
        guesses[4 * idx + 3] = self.dispersion

        limits.append((0, 0)) #Amplitude
        if self.linetype == 'absorption':
            limited.append((False, True)) #Amplitude
        if self.linetype == 'emission':
            limited.append((True, False)) #Amplitude
        if self.linetype == 'both':
            limited.append((False, False)) #Amplitude

        limits.append((line + llimits[0], line + llimits[1]))
        limited.append((True, True)) #Center

        limits.append((0, 0)) #width (gausssigma)
        limited.append((True, False)) #width

        limits.append((0, 0)) #width (lorentzsigma)
        limited.append((True, False)) #width

    if blends:
        blendlist = ascii.read(blends)

        for blendidx in range(len(blendlist)):

            # lprime = blendlist[blendidx]['lprime']
            # lsec = blendlist[blendidx]['lsec']

            if self.rv_sys == 0.:
                lprime = blendlist[blendidx]['lprime']
                lsec = blendlist[blendidx]['lsec']
            else:
                lprime = lambda_rv_shift(self,
                blendlist[blendidx]['lprime'])
                lsec = lambda_rv_shift(self,
                blendlist[blendidx]['lsec'])

            primeidx = np.where(lprime == guesses)[0]
            secidx = np.where(lsec == guesses)[0]

            if len(primeidx) == 1:

                self.logger.info('Blend detected: '\
                + str(blendlist[blendidx]['prime'])\
                + ' and ' + str(blendlist[blendidx]['sec']) + ' with ratio '\
                + str(blendlist[blendidx]['ratio']))

                guesses[secidx - 1] = blendlist[blendidx]['ratio']\
                * guesses[primeidx - 1]

                if lprime < lsec and (lprime + llimits[1]) > lsec + llimits[0]:
                    limits[secidx[0]]\
                    = (limits[primeidx[0]][1] + 0.01, limits[secidx[0]][1])

                    if limits[primeidx[0]][1] + 0.01 >= lsec:
                        limits[secidx[0]] = (lsec - 0.5, limits[secidx[0]][1])

                if lprime > lsec and (lprime + llimits[0]) < lsec + llimits[1]:
                    limits[secidx[0]] = (limits[secidx[0]][0],\
                                         limits[primeidx[0]][0] - 0.01)
                    if limits[primeidx[0]][0] - 0.01 <= lsec:
                        limits[secidx[0]] = (limits[secidx[0]][0], lsec + 0.5)

    return guesses, limits, limited


def update_parinfo(self, guesses, llimits, line_idx, blends,
                   parinfo, autoadjust, fwhm_block):

    '''
    Updates the parinfo file, created by pyspeckit.

    Args:
        guesses : :func:`numpy.array`
            The initial guesses for the the radial velocity fit guesses in
            the form [RV,sepctral_dispersion]

        llimits : :obj:`list`
            the limits for the wavelength fit as set in ``ppxf``

        line_idx : :obj:`str`
            Name of the primary line

       blends : :obj:`ascii`-file or :obj:`None`
           A file with primary lines that contain blends to provide a maximum
           amplitude ratio of the primary and the blend to prevent that the
           blend becomes the dominant line in the fit.

        parinfo: :obj:`dict`
            the parinfo file created by pyspeckit, which contains the fitted
            parameters for all input lines

        autoadjust : :obj:`bool`
            :obj:`True`: the wavelength limits ``llimit`` will be adjusted to
            the fit of the previous iteration. All other wavelength range are
            adjusted accordingly taking into account the proper velocity
            corrected shift :math:`\Delta \lambda/\lambda`. This is especially
            important to detect hyper-velocity stars.

            :obj:`False`: no adjustment to the limits done

        fwhm_block : :obj:`bool:obj:`
            :obj:`True`: The minimum fwhm of the voigt profiles of the fitted
            lines is the instrument's dispersion

            :obj:`False`: The minimum fwhm of the voigt profiles of the fitted
            lines is zero

    '''

    if self.rv_sys == 0.:
        lprime = self.cat.loc[line_idx, 'l_lab']
        primeidx = np.where(lprime == guesses)[0]
    else:
        lprime = lambda_rv_shift(self, self.cat.loc[line_idx, 'l_lab'])
        primeidx = np.where(lprime == guesses)[0]

    if autoadjust:
        lshift_in = parinfo[int(primeidx[0])]['value']\
        - guesses[int(primeidx[0])]

        for adj_idx in range(int(len(guesses) / 4.)):

            lshift = _lambda_shift(lshift_in, guesses[int(primeidx[0])],\
            guesses[int(4 * adj_idx + 1)])

            parinfo[int(4 * adj_idx + 1)]['limits']\
            = (guesses[int(4 * adj_idx + 1)] + llimits[0]\
            + lshift, guesses[int(4 * adj_idx + 1)]\
            + llimits[1] + lshift)

            parinfo[int(4 * adj_idx + 1)]['value']\
            = guesses[int(4 * adj_idx + 1)] + lshift

    #### check that fwhm is always > dispersion:

    if fwhm_block:
        for paridx in range(int(len(guesses) / 4.)):

            fwhm_g, fwhm_l, fwhm_v\
            = voigt_FWHM(np.array([parinfo[int(4 * paridx + 2)]['value']],\
            dtype=float), np.array([parinfo[int(4 * paridx + 3)]['value']],\
            dtype=float))
            if fwhm_v[0] < self.dispersion:
                target_fwhm_g = self.dispersion - fwhm_l[0]

                s = target_fwhm_g / 2. / np.sqrt(2. * np.log(2.))
                parinfo[int(4 * paridx + 2)]['limits']\
                = (s, parinfo[int(4 * paridx + 2)]['limits'][1])
                parinfo[int(4 * paridx + 2)]['value'] = s

    if blends:
        blendlist = ascii.read(blends)

        for blendidx in range(len(blendlist)):
            blendration = blendlist[blendidx]['ratio']
            lsec = blendlist[blendidx]['lsec']
            secidx = np.where(lsec == guesses)[0]
            if len(secidx) == 1:
                prime_profile = voigt_funct(self.spec_lambda_highres,\
                                    parinfo[int(primeidx[0])]['value'],\
                                    parinfo[int(primeidx[0] - 1)]['value'],\
                                    parinfo[int(primeidx[0] + 1)]['value'],\
                                    parinfo[int(primeidx[0] + 2)]['value'])

                sec_profile = voigt_funct(self.spec_lambda_highres,\
                                    parinfo[int(secidx[0])]['value'],\
                                    parinfo[int(secidx[0] - 1)]['value'],\
                                    parinfo[int(secidx[0] + 1)]['value'],\
                                    parinfo[int(secidx[0] + 2)]['value'])

                ind_max_prime = np.argmax(np.abs(prime_profile))
                ind_max_sec = np.argmax(np.abs(sec_profile))

                if np.max(np.abs(sec_profile)) >= \
                blendration * np.max(np.abs(prime_profile)):

                # if parinfo[int(secidx[0] - 1)]['value'] >= \
                # blendration * parinfo[int(primeidx[0] - 1)]['value']:

                    parinfo[int(secidx[0] - 1)]['limited'] = (True, True)

                    parinfo[int(secidx[0] - 1)]['limits'] \
                    = (blendration * parinfo[int(primeidx[0] - 1)]['value'], \
                    parinfo[int(secidx[0] - 1)]['limits'][1])

                    parinfo[int(secidx[0] - 1)]['value']\
                    = 0.99 * blendration * prime_profile[ind_max_prime]
                    # = 0.99 * blendration\
                    # * parinfo[int(primeidx[0] - 1)]['value']

                if parinfo[int(primeidx[0] - 1)]['value'] == 0.:
                    parinfo[int(secidx[0] - 1)]['limits'] \
                    = (0.01 * (-1), \
                    parinfo[int(secidx[0] - 1)]['limits'][1])
                    parinfo[int(secidx[0] - 1)]['value']\
                    = blendration * (0.01) * (-1.)

    return parinfo


def plotcolor(n):
    colarray = np.array(['Lime', 'Blue', 'Magenta', 'Olive', 'Maroon',
    'indigo', 'orange', 'Cyan', 'Yellow', 'Silver'])
    colselect = colarray[n % len(colarray)]
    return colselect


def baseline_extract(self, specfit, poly_order):
    coeff = np.zeros(poly_order + 1)
    power = np.arange(poly_order + 1)
    for order in range(poly_order + 1):
        coeff[order] = specfit.header['BLCOEF' + str('{:02d}'.format(order))]
    baseline = np.poly1d(coeff / (self.specbinsize ** power[::-1]))
    return baseline


def voigt_FWHM(sigma_g, gamma_l):
    #calculating the FWHM of the voigt, gauss, and lorentz profile
    c_0 = 2.0056
    c_1 = 1.0593
    fwhm_g = 2. * np.sqrt(2. * np.log(2.)) * sigma_g
    fwhm_l = 2. * gamma_l
    phi = fwhm_l / fwhm_g
    fwhm_v = np.zeros_like(sigma_g, dtype=float)
    for i in range(len(fwhm_g)):
        if fwhm_g[i] == 0:
            fwhm_v[i] = fwhm_l[i]
        else:
            fwhm_v[i] = fwhm_g[i] * (1. - c_0 * c_1 + np.sqrt(phi[i] ** 2\
            + 2. * c_1 * phi[i] + (c_0 * c_1) ** 2))

    return fwhm_g, fwhm_l, fwhm_v


def voigt_funct(x_array, x_cen, amplitude, sigma, gamma):

    if sigma > 0 and gamma > 0:
        z = ((x_array - x_cen) + 1j * gamma) / (sigma * np.sqrt(2.))
        y_arr = amplitude * np.real(wofz(z))
    if sigma == 0:
        y_arr = _lorentzian(x_array, x_cen, amplitude, gamma)
    if gamma == 0:
        y_arr = _gaussian(x_array, x_cen, amplitude, sigma)

    return y_arr


def spec_res_downgrade(l_in, spec_in, l_out):
    templ_spec = spectrum.ArraySourceSpectrum(wave=l_in, flux=spec_in)
    white_filter = spectrum.ArraySpectralElement(l_out,\
    np.ones(len(l_out)), waveunits='angstrom')
    convolved_spec = observation.Observation(templ_spec,\
    white_filter, binset=l_out, force='taper').binflux

    return convolved_spec


def line_clipping(self, x, line_significants, sigma=3):

    mask = np.zeros_like(x, dtype=int)
    ind_sig = np.where((self.cat.loc[:,\
    'significance'].values.astype(np.float64) < line_significants)\
                       | (self.cat.loc[:, 'used'].values == 'f'))
    mask[ind_sig] = 1

    x_red1 = np.delete(x, np.where(mask == 1))

    if len(x_red1) >= 3:
        if len(x_red1) == 3:
            median = np.median(x_red1)
            mad = MAD(x_red1)
            ind = np.where((x < median - sigma * mad) | (x > median\
            + sigma * mad))[0]
        if len(x) > 3:
            gr = np.array(list(inter_comb(x_red1, len(x_red1) - 2)))

            std_gr = [np.std(it) for it in gr]
            x_red = gr[np.argmin(std_gr)]

            median = np.median(x_red)
            mad = MAD(x_red)
            ind = np.where((x < median - sigma * mad) | (x > median\
            + sigma * mad))[0]

        mask[ind] = 1
    x_masked = np.ma.masked_array(x, mask=[mask])
    return x_masked


def clipped_MAD(x):
    if len(x) <= 3:
        mad = MAD(x)
    if len(x) > 3:
        x = np.delete(x, [np.argmin(x), np.argmax(x)])
        mad = MAD(x)
    return mad


def continuum_deviation(self, l_in, f_in, baseline, contorder):

    if self.linetype == 'absorption':
        peaks = find_peaks(f_in * (-1), width=2)[0]
    else:
        peaks = find_peaks(f_in, width=2)[0]
    mask = np.zeros_like(f_in)

    for p in peaks:
        mask[p - 3:p + 4] = 1

    remove = ~np.ma.array(l_in, mask=mask).mask

    l_masked = l_in[remove]
    f_masked = f_in[remove]
    baseline_masked = baseline[remove]

    masked_fit = np.polyfit(l_masked, f_masked, contorder)
    masked_fitfunc = np.poly1d(masked_fit)

    cont_dev = MAD(f_masked / np.median(f_masked)
    - baseline_masked / np.median(baseline_masked))

    return cont_dev


def ABtoVega(instrument, bandpass):

    bp = ObsBandpass(str(instrument) + ',wfc1,'\
    + str(bandpass) + ',mjd#57754')
    spec_bb = BlackBody(10000)
    spec_bb_norm = spec_bb.renorm(1, 'counts', bp)
    obs = observation.Observation(spec_bb_norm, bp)

    # Get photometric calibration information.
    photflam = obs.effstim('flam')
    photplam = bp.pivot()

    zp_vega = obs.effstim('vegamag')
    zp_st = obs.effstim('stmag')
    zp_ab = obs.effstim('abmag')

    difference = zp_vega - zp_ab

    return difference


def lambda_rv_shift(self, lam):
    lambda_new = lam + self.rv_sys * lam / const.c.to('km/s').value
    if self.rv_sys == 0:
        lambda_new = lam
    return lambda_new


def _lambda_shift(dlin, lin, lout):
    '''
       This functions shifts the wavelength limits if the central wavelength
       is shifted. This assures that lambda never runs out of bounds

    '''

    dlout = dlin * lout / lin
    return dlout


def _gaussian(x, mu, A, sigma):
    prefact = 1. / np.sqrt(2. * np.pi * sigma ** 2)
    exponetial = np.exp((-1.) * ((x - mu) ** 2) / (2. * sigma ** 2))
    returnfunction = A * exponetial
    return returnfunction


def _lorentzian(x, mu, A, gamma):
    returnfunction = A * gamma / np.pi / ((x - mu) ** 2 + gamma ** 2)
    return returnfunction