# -*- coding: utf-8 -*-
#Christiaan Dik and Stan Barmentloo 2020
#Bury Unwanted Trends to Correct and Horizontalize Every lightcuRve (BUTCHER)

import numpy as np

def mag_to_flux(mag):
    """
    Converts a given array of magnitude values to normalized flux values.
    
    Parameters
    ----------
    mag : array-like
        The magnitude data.
    
    Returns
    -------
    flux : array-like
        The normalized flux data
    """
    flux = 10**(-0.4*mag)/np.mean(10**(-0.4*mag))
    return flux

def emag_to_eflux(mag, emag):
    """
    Converts a given array of uncertainty on magnitudes to normalized uncertainty on flux.
    
    Parameters
    ----------
    mag : array-like
        The magnitude data.
    emag : array-like
        The errors on the magnitude data. 
        Needs to have the same shape as mag.
        
    Returns
    -------
    eflux : array-like
        The error on the flux, normalized.
    """
    eflux = np.sqrt((-0.4*np.log(10)*np.exp(-0.4*np.log(10)*mag)*emag)**2)/np.mean(10**(-0.4*mag))
    return eflux


def long_correct(time, flux, eflux):
    """
    Fits a line to the data and removes this trend.
    
    Parameters
    ----------
    time : array-like
        The temporal data of the flux measurements.
    flux : array-like
        The flux data.
        Needs to have the same shape as time.
    eflux : array-like
        The error on the flux data.
        Need to have the same shape as time and flux.
        
    Returns
    -------
    corr_flux : array-like
        The flux data with a linear correction.
    """
    coef, cov = np.polyfit(time, flux, 1, w=1/(eflux**2), cov=True)
    corr_flux = flux/(coef[0]*time + coef[1])
    
    corr_flux/=np.mean(corr_flux)
    
    return corr_flux
    



