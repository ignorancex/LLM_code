from astropy.io import fits
import pandas as pd
import glob
import os
import numpy as np

def flux_to_lum(flux, distance):
    """Distance should be in parsecs, flux in erg cm^-2 sec^-1"""
    dist_in_cm = 3.086*10**18.0 * distance # cm
    lum = np.multiply(flux, 4 * np.pi * np.power(dist_in_cm, 2.0))
    return lum

def planet_flux(luminosity, distance):
    """Distance should be in AU, luminosity in erg sec^-1"""
    dist_in_cm = (6.685*10**(-14.))**-1.0 * distance # cm /au * cm3.086*10**18.0 * distance # cm
    flux_on_planet = np.true_divide(luminosity, np.multiply(4 * np.pi, np.power(dist_in_cm, 2.0)))
    return flux_on_planet


def tidal_evo(y, t, Q, r, per, dmdtval):
    """Units: earth masses, years, AU
    """
    k = 0.3 # some constant in tidal equations (love number) - doesn't matter if we plot everything in terms of Q' in the end
    G = 0.000118413597733711048158640226628895184135977337110481586 # (*AU^3/(earth mass*year^2),
                                                                # which is G in these weird units*)

    mstar = 27000.0 # mass of trappist-1 star, in earth masses
    rstar = 5.627e-4 # radius of trappist-1 star, in AU
    # also all angles must be in radians
    a,e,m = y
    signval = np.sign(2 * (2*np.pi / 3.3) - 3*(2*np.pi / per))
    qprime_p = Q / ((2/3.) * k)
    qprime_s = 1e5 / ((2/3.) * k) # https://iopscience.iop.org/article/10.3847/1538-4357/aad40e/pdf

    dadt =  (signval * 9./2. * np.sqrt(G/ mstar) * np.power(rstar, 5.) * m * np.power(qprime_s, -1.) - 63./2. * np.multiply(qprime_p, m)**-1*np.sqrt(G * mstar**3.0) *np.power(r,5.0) * np.power(e, 2.0)) * np.power(a, -11/2.) 
    dedt =  (signval * 171./16. * np.sqrt(G/ mstar) * np.power(rstar, 5.) * m * np.power(qprime_s, -1.) - 63./4. * np.multiply(qprime_p, m)**-1*np.sqrt((G * mstar**3.0)) *(r)**5.0) * np.power(a, -13/2.) * e
    #drdt = -1e-18 / (4.0 * np.pi * r**2.) # dm/dt / (4 pi r^2) = dr/dt
    dmdt =  -1 * dmdtval
    dydt = np.asarray([dadt, dedt, dmdt])
    return dydt