import numpy as np
from astropy import units
from mpmath import expint, mp, fp
from scipy.optimize import curve_fit
from astropy import constants as const

__all__ = ['get_energy_limits', 'planck', 'fit_ffd', 'linear', 'logpdf']

def planck(lam, T, rad):
    """
    Planck function.
    """
    result = 2.*np.pi*const.h*(const.c**2)/(lam**5)
    exp = (const.h*const.c/(lam*const.k_B*T))#.to(u.m/u.m)
    result /= (np.exp(exp) - 1.0)

    result *= (lam[1]-lam[0])
    result = result.to(units.erg/units.s/units.m**2) * (rad.to(units.m))**2
    return result


def get_energy_limits(tab, temp_ranges, low_ed=60, low_amp=0.005,
                      flare_temp=9000.0):
    """
    Defines the lower limit for what flare energy to fit for a given stellar
    effective temperature range.

    Parameters
    ----------
    tab : astropy.table.Table
       Table of flares.
    temp_ranges : np.ndarray
       Array of temperature ranges to evaluate over. Array should be structured as:
       `temp_ranges = [ [lowlim1, upplim1], [lowlim2, upplim2] ]`.
    low_ed : float, optional
       The lower-limit flare equivalent duration to trust in units of seconds.
       Default is 60.
    low_amp : float, optional
       The lower-limit flare amplitude to trust. Defualt is 0.005 (0.5% amplitude).
    flare_temp : float, optional
       The flare temperature to assume. Default is 9000K.
    """
    TEMP_LIMITS = []

    lam = np.arange(600,1000) * units.nm # TESS bandpass

    for z in range(len(temp_ranges)):

        if z == 0:
            minrow = tab[(tab['teff']>0) & (tab['lum']>0)]
            minrow.sort('teff')
            minrow = minrow[0]
            lum = minrow['lum']
            rad = minrow['rad']
            real_teff = minrow['teff']
            star = planck(lam, real_teff*units.K, rad*units.Rsun).to(units.erg/units.s)
            flare = planck(lam, flare_temp*units.K,  rad*units.Rsun).to(units.erg/units.s)
            scale = np.nansum(star.value) / np.nansum(flare.value)

            TEMP_LIMITS.append((low_amp * (low_ed*units.s) * \
                               (lum * units.Lsun) * scale).to(units.erg).value)

        rows = tab[(tab['teff']>=temp_ranges[z][0]) & (tab['lum']>0) &
                   (tab['teff']<temp_ranges[z][1])]

        minrow = rows[rows['teff']>=np.nanmedian(rows['teff'])]
        minrow.sort('teff')

        rad = np.nanmedian(minrow['rad'])
        arg = np.where(minrow['rad']>=rad)[0][0]

        lum = minrow['lum'][arg]
        rad = minrow['rad'][arg]
        real_teff = minrow['teff'][arg]

        star = planck(lam, real_teff*units.K, rad*units.Rsun).to(units.erg/units.s)
        flare = planck(lam, flare_temp*units.K,  rad*units.Rsun).to(units.erg/units.s)
        scale = np.nansum(star.value) / np.nansum(flare.value)

        TEMP_LIMITS.append((low_amp * (low_ed*units.s) * (lum * units.Lsun) * \
                            scale).to(units.erg).value)

    return TEMP_LIMITS


def linear(x, m, b):
    """ A line. """
    return m*x+b


def fit_ffd(data, limit, yerr=None):
    """
    Gives an initial guess for the FFD slope (in units of energy).

    Parameters
    ----------
    data : tuple
       The output of `plt.hist`, where the first index is an array of bins,
       the second index is an array of the bin edges, and the third index is a
       `matplotlib.container.BarContainer` object.
    limit : float
       Energy limit to fit to.
    yerr : np.array, optional
       Errors on the flare frequency distribution bins.

    Returns
    --------
    popt : np.array
       Best-fit slope and offset.
    perr : np.array
       Error on the slope and offset.
    data : np.array
       The final data fit to, after some masking. `x = data[:,0]`; `y = data[:,1]`;
       `yerr = data[:,2]`.
    """
    x = (data[1][1:] + data[1][:-1])/2.0
    n = data[0]

    mask = (x != 0) & (x > limit) & (n != 0)
    logx = np.log10(x[mask]) + 0.0
    logn = np.log10(n[mask]) + 0.0

    if yerr is None:
        yerr=np.sqrt(n[mask])
    else:
        yerr = (yerr[mask]/n[mask]) * (1.0/np.log(10))
        yerr[yerr==0] = 1e-5

    popt, pcov = curve_fit(linear, logx, logn, sigma=yerr,
                           p0=[-0.8, 22],
                           bounds=([-2.0, -50.0],
                                   [0.0, 50.0]),
                           maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))

    data = np.array([logx, logn, np.sqrt(n[mask])])

    return popt, perr, data.T


# Functional form of PDF with respect to log_10 a
def logpdf(loga, q, astar, amp_min):
    a = 10.**loga
    norm = amp_min**(q-1) / fp.mpf(expint(q, amp_min/astar)) * np.log(10)
    return norm * a**-(q-1) * np.exp(-a/astar)
