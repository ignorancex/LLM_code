# -*- coding: utf-8 -*-
"""
Various functions to calculate the Compton wavelength and related quantities.

Created: June 2022
Author: Bradley March
"""

# %% Python Preamble

# Import relevant modules
import numpy as np

# Import user created modules
import Packages.fR_functions as fRf
from constants import omega_L, omega_m, H0, c, R0

# %% Compton wavelength calculations


def calc_full_Compton_wavelength(fR: np.ndarray, fR0: float) -> np.ndarray:
    """
    Calculates the Compton wavelength of the fR field, 
    including subdominant terms.

    Parameters:
    fR (numpy array): The values of the fR scalar field.
    fR0 (float): The background value of the fR scalar field.

    Returns:
    numpy array: The Compton wavelength, with the same shape as fR.
    """
    # calculate f(R) model parameters
    a = - 4 * omega_L**2 / (omega_m + 4 * omega_L)**2 * (1 / fR0)
    m2 = - 3 * H0**2 * (omega_m + 4 * omega_L)**2 / (2 * omega_L * c**2) * fR0
    # calculate R, f(R) & higher order derivatives
    R = m2 * (np.sqrt(- a / fR) - 1)
    f = - a * m2 * R / (R + m2)
    fRR = - 2 * fR / (R + m2)
    # calculate V''(phi) (equivalent to 'mass^2' of field)
    Vpp = 1 / fRR - (3*R + 4*f - R*fR) / (1 + fR)
    Vpp = Vpp / 3
    # return Compton wavelength
    Lc = np.sqrt(1/Vpp)
    return Lc


def calc_Compton_wavelength(fR: np.ndarray, fR0: float) -> np.ndarray:
    """
    Calculates the Compton wavelength of the fR field. 
    Simplifies the scalar mass, neglecting subdominant terms.

    Parameters:
    fR (numpy array): The values of the fR scalar field.
    fR0 (float): The background value of the fR scalar field.

    Returns:
    numpy array: The Compton wavelength, with the same shape as fR.
    """
    # calculate constant prefactor
    prefactor = (c / H0) * np.sqrt(2 / (omega_m + 4 * omega_L))
    # calculate field dependent factor
    ff = fR**3 / fR0
    # return Compton wavelength
    Lc = prefactor * ff**(1/4)
    return Lc


def calc_background_Compton_wavelength(fR0: float) -> float:
    """
    Calculates background Compton wavelength for given fR0 parameter(s).

    Parameters:
    fR0 (float): The background value of the fR scalar field.

    Returns:
    float: The background value of the Compton wavelength.
    """
    Lc = (c / H0) * np.sqrt(2 * np.abs(fR0) / (omega_m + 4 * omega_L))
    return Lc


# %% Screened fR field and Compton wavelength


def calc_screened_fR(fR0: float, drho: np.ndarray) -> np.ndarray:
    """
    Calculates the screened fR field. 
    Assumes lap(fR)=0 and analytically calculates fR from EoM.

    Parameters:
    fR0 (float): The background value of the fR scalar field.
    drho (numpy array): The density perturbation.

    Returns:
    numpy array: The screened fR field, with the same shape as drho.
    """
    drho_term = fRf.delta_rho_term(drho)
    fR_scr = fR0 * (1 + drho_term / R0)**(-2)
    return fR_scr


def calc_screened_Compton_wavelength(fR0: float, drho: np.ndarray
                                     ) -> np.ndarray:
    """
    Calculates the screened Compton wavelength.
    Assumes lap(fR)=0 and analytically calculates fR from EoM.

    Parameters:
    fR0 (float): The background value of the fR scalar field.
    drho (numpy array): The density perturbation.

    Returns:
    numpy array: The screened Compton wavelength, 
        with the same shape as fR.
    """
    fR_scr = calc_screened_fR(fR0, drho)
    Lc_scr = calc_Compton_wavelength(fR_scr, fR0)
    return Lc_scr


# %% Stellar separation and unrelaxed field calculations


def calc_mean_separation(rho: np.ndarray, Mp: float) -> np.ndarray:
    """
    Calculates the typical point particle separation in a 
    density profile, rho, for an average particle mass, Mp.

    Parameters:
    rho (numpy array): The density profile.
    Mp (float): The average particle mass.

    Returns:
    numpy array: The typical point particle separation, with the same 
    shape as rho.
    """
    separation = np.ma.divide(Mp**(1/3), rho**(1/3))
    return separation


def calc_unrelaxed_fR(fR0: float, S: np.ndarray) -> np.ndarray:
    """
    Calculates the expected unrelaxed field fR.
    I.e. calculate the value of fR that results in a Compton wavelength 
    equal to the separation, S.

    Parameters:
    fR0 (float): The background value of the fR scalar field.
    S (numpy array): The separation.

    Returns:
    numpy array: The unrelaxed field fR, with the same shape as S.
    """
    prefactor = c / H0 * np.sqrt(2 / (omega_m + 4*omega_L))
    fR = prefactor**(-4/3) * (-np.abs(fR0)**(1/3)) * S**(4/3)
    return fR
