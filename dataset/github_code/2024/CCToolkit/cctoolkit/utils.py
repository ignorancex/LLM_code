"""
utils.py

This module can contain utility functions or constants that may be useful
across the package. 
"""
import numpy as np

def w_tophat(r, k):
    """
    Fourier Transform of the spherical top-hat function.

    Parameters:
    -----------
    r : float
        Radius of the spherical top-hat window.
    k : float or np.ndarray
        Wavenumber(s).

    Returns:
    --------
    float or np.ndarray
        The Fourier Transform of the top-hat function.
    """
    return (3. * (np.sin(k * r) - k * r * np.cos(k * r)) / ((k * r) ** 3))

def dw_tophat_sq_dr(r, k):
    """
    Derivative of the square of the Fourier Transform of the top-hat function with respect to r.

    Parameters:
    -----------
    r : float
        Radius of the spherical top-hat window.
    k : float or np.ndarray
        Wavenumber(s).

    Returns:
    --------
    float or np.ndarray
        The derivative with respect to r.
    """
    return (-18. * (k * r * np.cos(k * r) - np.sin(k * r)) * 
            (3. * k * r * np.cos(k * r) + (-3. + (k ** 2) * (r ** 2)) * np.sin(k * r))) / ((k ** 6) * (r ** 7))

def s_rsq_integrand_given_Dk(r, k, Dk):
    """
    Integrand for the calculation of sigma_r^2.

    Parameters:
    -----------
    r : float
        Radius of the spherical top-hat window.
    k : np.ndarray
        Wavenumber array.
    Dk : np.ndarray
        Dimensionless power spectrum array.

    Returns:
    --------
    np.ndarray
        The integrand for sigma_r^2.
    """
    return ((w_tophat(r, k)) ** 2) * Dk / k

def d_s_rsq_integrand_given_Dk(r, k, Dk):
    """
    Integrand for the derivative of sigma_r^2 with respect to r.

    Parameters:
    -----------
    r : float
        Radius of the spherical top-hat window.
    k : np.ndarray
        Wavenumber array.
    Dk : np.ndarray
        Dimensionless power spectrum array.

    Returns:
    --------
    np.ndarray
        The integrand for the derivative of sigma_r^2 with respect to r.
    """
    return (dw_tophat_sq_dr(r, k)) * Dk / k

def ssq_given_Dk(r, k, Dk):
    """
    Compute sigma_r^2 for a given radius and power spectrum.

    Parameters:
    -----------
    r : float
        Radius of the spherical top-hat window.
    k : np.ndarray
        Wavenumber array.
    Dk : np.ndarray
        Dimensionless power spectrum array.

    Returns:
    --------
    float
        The value of sigma_r^2.
    """
    s_sq = np.trapz(s_rsq_integrand_given_Dk(r, k, Dk), x=k)
    return s_sq

def d_s_given_Dk(r, k, Dk):
    """
    Compute the derivative of sigma_r with respect to r.

    Parameters:
    -----------
    r : float
        Radius of the spherical top-hat window.
    k : np.ndarray
        Wavenumber array.
    Dk : np.ndarray
        Dimensionless power spectrum array.

    Returns:
    --------
    float
        The derivative of sigma_r with respect to r.
    """
    d_s = np.trapz(d_s_rsq_integrand_given_Dk(r, k, Dk), x=k)
    return d_s / 2.0 / np.sqrt(ssq_given_Dk(r, k, Dk))

def compute_sigma8_norm(k, Dk, sigma8):
    """
    Normalize the dimensionless power spectrum to the given sigma8.

    Parameters:
    -----------
    k : np.ndarray
        Wavenumber array.
    Dk : np.ndarray
        Dimensionless power spectrum array.
    sigma8 : float
        The value of sigma8.

    Returns:
    --------
    float
        The normalization factor for the power spectrum.
    """
    s_sq = ssq_given_Dk(8, k, Dk)
    return s_sq / sigma8 ** 2

def delta_c(Om, z=None):
    """
    Calculate the critical density contrast (delta_c) using eq. A.6 of Kitayama & Suto.

    Parameters:
    -----------
    Om : float or callable
        Matter density parameter. If callable, it should return Om as a function of z.
    z : float or array-like, optional
        Redshift at which to evaluate Om if Om is a function.

    Returns:
    --------
    float
        The critical density contrast delta_c.
    """
    if z is None:
        return 3.0 / 20.0 * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.012299 * np.log10(Om))
    else:
        return 3.0 / 20.0 * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.012299 * np.log10(Om(z)))

def virial_Delta(Om, z=None):
    """
    Calculate the virial overdensity (Delta_vir) in critical density unit using Bryan and Norman fit.

    Parameters:
    -----------
    Om : float or callable
        Matter density parameter. If callable, it should return Om as a function of z.
    z : float or array-like, optional
        Redshift at which to evaluate Om if Om is a function.

    Returns:
    --------
    float
        The virial overdensity.
    """
    if z is None:
        x = Om - 1.0
    else:
        x = Om(z) - 1

    return 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

critical_density0 = 277536627245.708 # Msun/h / (Mpc/h)**3
    