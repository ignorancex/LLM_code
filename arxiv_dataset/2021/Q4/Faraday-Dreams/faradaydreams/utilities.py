#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def get_polarization_and_weights(P_q, P_u, error_q, error_u):
    """  
    Inputs:
    P_q: Stokes Q spectra
    P_u: Stokes U spectra
    error_q: Stokes Q error
    error_u: Stokes U error

    Returns:
    y: Q + iU spectra
    weights: weights for y to use when fitting data (in units of sigma rather than Jy)
    sigma: the noise level estimated as the RMS of error_q and error_u
    """
    y = np.array(P_q) + 1j * np.array(P_u)
    weights_q = np.array(error_q)
    weights_u = np.array(error_u)
    sigma = np.mean(np.abs(weights_q + 1j * weights_u))
    weights = sigma / np.abs(weights_q + 1j * weights_u)
    return y, weights, sigma


def calculate_lambda2(freq, dfreq):
    c = 299792458.  #speed of light m/s
    lambda2 = (c / freq)**2  #wavelength^2 coverage
    if dfreq is None:
        return lambda2, None
    dlambda2 = (c / (freq - dfreq / 2.))**2 - (
        c / (freq + dfreq / 2.))**2  #channel width in wavelength squared.
    return lambda2, dlambda2


def convolve_solution(solution, phi, sigma_phi):
    return np.convolve(solution, np.exp(-phi[::-1]**2 / (2 * sigma_phi**2)),
                       'same')
