# -*- coding: utf-8 -*-
"""
Relations between galactic density model parameters, plus functions to run the 
pipeline and calculates a typical galaxy from a single input, Mvir. 


Created: June 2022
Author: Bradley March
"""
# %% Preamble
import numpy as np
from constants import pc, kpc, Mpc, h, M_sun, rho_c

# define galaxy profile constants
r_min, r_max = 50 * pc, 10 * Mpc  # inner/outer grid cutoff
splashback_factor = 2.2  # outer density cutoff
delta = 200  # virial density contrast, i.e. Mvir = M200c

# %% Stellar-Halo Mass Relations


def virial_mass_to_stellar_mass(virial_mass: float,
                                logM1: float = 11.590, N: float = 0.0351,
                                beta: float = 1.376, gamma: float = 0.608
                                ) -> float:
    """
    Stellar-Halo Mass Relation (SHMR). 
    [Equation 2 in Moster, Naab and White (2012), 
    parameters from Table 3.]

    Parameters:
    virial_mass (float): The virial mass of the halo.
    logM1 (float): The logarithm of the characteristic mass.
    N (float): The normalisation factor.
    beta (float): The low-mass slope.
    gamma (float): The high-mass slope.

    Returns:
    float: The stellar mass of the galaxy.
    """
    M1 = M_sun * 10**(logM1)
    numerator = 2 * N * virial_mass
    denominator = (virial_mass / M1)**(-beta) + (virial_mass / M1)**(gamma)
    stellar_mass = numerator / denominator
    return stellar_mass


# %% Dark matter parameter pipeline functions


def calc_concentration_parameter(virial_mass: float) -> float:
    """
    Empirical mass-concentration relation.
    [Equation 8 of Dutton and Maccio (2014).]

    Parameters:
    virial_mass (float): The virial mass of the halo.

    Returns:
    float: The concentration parameter of the NFW profile.
    """
    concentration_parameter = 10**(0.905 - 0.101 *
                                   np.log10(virial_mass * h/(M_sun * 1e12)))
    return concentration_parameter


def calc_virial_radius(virial_mass: float) -> float:
    """
    From definition of virial mass.

    Parameters:
    virial_mass (float): The virial mass of the halo.

    Returns:
    float: The virial radius of the halo.
    """
    virial_radius = (3 * virial_mass / (4 * np.pi * delta * rho_c))**(1/3)
    return virial_radius


def calc_virial_scale_radius(virial_radius: float,
                             concentration_parameter: float
                             ) -> float:
    """
    From definition of concentration parameter.

    Parameters:
    virial_radius (float): The virial radius of the halo.
    concentration_parameter (float): The concentration parameter of 
        the NFW profile.

    Returns:
    float: The scale radius of the NFW profile.
    """
    virial_scale_radius = virial_radius / concentration_parameter
    return virial_scale_radius


def calc_virial_normalisation(virial_mass: float, virial_radius: float,
                              concentration_parameter: float) -> float:
    """
    Calculates the NFW density normalisation constant by reversing the 
    relation for the enclosed mass at a given radius.

    Parameters:
    virial_mass (float): The virial mass of the halo.
    virial_radius (float): The virial radius of the halo.
    concentration_parameter (float): The concentration parameter of 
        the NFW profile.

    Returns:
    float: The normalisation constant of the NFW profile.
    """
    virial_scale_radius = calc_virial_scale_radius(
        virial_radius, concentration_parameter)
    denom = np.log(1 + concentration_parameter) - \
        (concentration_parameter / (1 + concentration_parameter))
    denom = 4 * np.pi * virial_scale_radius**3 * denom
    virial_normalisation = virial_mass / denom
    return virial_normalisation


# %% Stellar parameter pipeline functions


def calc_stellar_scale_length(stellar_mass: float,
                              alpha: float = 0.14, beta: float = 0.39,
                              gamma: float = 0.10, M_0: float = 3.98e10*M_sun
                              ) -> float:
    """
    Calculate stellar disc scale length by passing from mass to 
    half-light radius 
    [Equation 18 in Shen et al. 2003, parameters from Table 1.]
    and then analytically relating the half-light radius to the radius 
    where half the stellar mass is enclosed.

    Parameters:
    stellar_mass (float): The stellar mass of the galaxy.
    alpha (float): The low-mass slope.
    beta (float): The high-mass slope.
    gamma (float): The normalisation factor.
    M_0 (float): The characteristic mass.

    Returns:
    float: The scale length of the stellar disc.
    """
    # calculate half-light radius
    R_hl = gamma * (stellar_mass / M_sun)**(alpha) \
        * (1 + stellar_mass / M_0)**(beta - alpha)
    # convert to stellar radial scale length
    stellar_scale_length = 0.595824 * R_hl * kpc
    return stellar_scale_length


def calc_stellar_normalisation(stellar_mass: float, stellar_scale_length: float
                               ) -> float:
    """
    Reversing the total stellar mass definition to find the 
    (surface) density normalisation.

    Parameters:
    stellar_mass (float): The stellar mass of the galaxy.
    stellar_scale_length (float): The scale length of the stellar disc.

    Returns:
    float: The normalisation constant of the stellar disc profile.
    """
    denom = 2 * np.pi * stellar_scale_length**2
    stellar_normalisation = stellar_mass / denom
    return stellar_normalisation


def calc_stellar_scale_height(stellar_scale_length: float) -> float:
    """
    Empirical relation between scale length and height.
    [Equation 1 of Bershady et al. (2010).]

    Parameters:
    stellar_scale_length (float): The scale length of the stellar disc.

    Returns:
    float: The scale height of the stellar disc.
    """
    exponent = 0.367 * np.log10(stellar_scale_length / kpc) + 0.708
    stellar_scale_height = stellar_scale_length * 10**(-exponent)
    return stellar_scale_height


# %% Calculate density profiles from parameters


def calc_dark_matter_density(rgrid: np.ndarray, virial_normalisation: float,
                             virial_scale_radius: float) -> np.ndarray:
    """
    Calculate the dark matter density as an NFW profile.
    [Navarro, Frenk and White (1995).]

    Parameters:
    rgrid (numpy array): The radial grid.
    virial_normalisation (float): The normalisation constant of 
        the NFW profile.
    virial_scale_radius (float): The scale radius of the NFW profile.

    Returns:
    numpy array: The dark matter density profile.
    """
    x = rgrid / virial_scale_radius
    dark_matter_density = virial_normalisation / (x * (1 + x)**2)
    return dark_matter_density


def calc_stellar_disc_density(rgrid: np.ndarray, thgrid: np.ndarray,
                              stellar_normalisation: float,
                              stellar_scale_length: float,
                              stellar_scale_height: float) -> np.ndarray:
    """
    Calculate the stellar disc density as a double-exponential profile.
    [Binney and Tremaine (2008).]

    Parameters:
    rgrid (numpy array): The radial grid.
    thgrid (numpy array): The angular grid.
    stellar_normalisation (float): The normalisation constant of 
        the stellar disc profile.
    stellar_scale_length (float): The scale length of the stellar disc.
    stellar_scale_height (float): The scale height of the stellar disc.

    Returns:
    numpy array: The stellar disc density profile.    
    """
    # set up polar coordinates
    zgrid = np.abs(rgrid * np.cos(thgrid))
    Rgrid = rgrid * np.sin(thgrid)
    # calculate density
    prefactor = stellar_normalisation / (2 * stellar_scale_height)
    expR = np.exp(-Rgrid / stellar_scale_length)
    expz = np.exp(-zgrid / stellar_scale_height)
    stellar_disc_density = prefactor * expR * expz
    return stellar_disc_density


# %% Derive density profile parameters from input virial_mass


def get_dark_matter_parameters(virial_mass: float) -> dict:
    """
    Runs the pipeline of empirical and analytic relations to derive all 
    dark matter density profile parameters, assuming no scatter on relations.

    Parameters:
    virial_mass (float): The virial mass of the halo.

    Returns:
    dict: A dictionary of dark matter density profile parameters.
    """
    concentration_parameter = calc_concentration_parameter(virial_mass)
    virial_radius = calc_virial_radius(virial_mass)
    virial_scale_radius = calc_virial_scale_radius(
        virial_radius, concentration_parameter)
    virial_normalisation = calc_virial_normalisation(
        virial_mass, virial_radius, concentration_parameter)

    # create dictionary for Dark Matter Parameters
    DMP_dict = {'M': virial_mass,
                'c': concentration_parameter,
                'R': virial_radius,
                'norm': virial_normalisation,
                'Rs': virial_scale_radius,
                'SB': splashback_factor * virial_radius}

    return DMP_dict


def get_stellar_disc_parameters(virial_mass: float) -> dict:
    """
    Runs the pipeline of empirical and analytic relations to derive all 
    stellar disc density profile parameters, assuming no scatter on relations.

    Parameters:
    virial_mass (float): The virial mass of the halo.

    Returns:
    dict: A dictionary of stellar disc density profile parameters.
    """
    stellar_mass = virial_mass_to_stellar_mass(virial_mass)
    stellar_scale_length = calc_stellar_scale_length(stellar_mass)
    stellar_normalisation = calc_stellar_normalisation(
        stellar_mass, stellar_scale_length)
    stellar_scale_height = calc_stellar_scale_height(stellar_scale_length)

    # create dictionary for Stellar Disc Parameters
    SDP_dict = {'m': stellar_mass,
                'R': stellar_scale_length,
                'z': stellar_scale_height,
                'norm': stellar_normalisation}

    return SDP_dict


# %% Density pipeline with virial_mass as an input


def get_dark_matter_density(virial_mass: float, rgrid: np.ndarray,
                            splashback_cutoff: bool = True) -> np.ndarray:
    """
    Calculates the dark matter density profile for a given mass, assuming
    mean values on all empirical relations.

    Parameters:
    virial_mass (float): The virial mass of the halo.
    rgrid (numpy array): The radial grid.
    splashback_cutoff (bool): Whether to cut off the density at 
        the splashback radius.

    Returns:
    numpy array: The dark matter density profile.
    """
    DMP = get_dark_matter_parameters(virial_mass)
    dark_matter_density = calc_dark_matter_density(
        rgrid, DMP['norm'], DMP['Rs'])
    if splashback_cutoff:  # cut off density at SB radius
        dark_matter_density[rgrid >= DMP['SB']] = 0
    return dark_matter_density


def get_stellar_disc_density(virial_mass: float,
                             rgrid: np.ndarray, thgrid: np.ndarray,
                             splashback_cutoff: bool = True) -> np.ndarray:
    """
    Calculates the stellar disc density profile for a given mass, assuming
    mean values on all empirical relations.

    Parameters:
    virial_mass (float): The virial mass of the halo.
    rgrid (numpy array): The radial grid.
    thgrid (numpy array): The angular grid.
    splashback_cutoff (bool): Whether to cut off the density at 
        the splashback radius.

    Returns:
    numpy array: The stellar disc density profile.
    """
    SDP = get_stellar_disc_parameters(virial_mass)
    stellar_disc_density = calc_stellar_disc_density(
        rgrid, thgrid, SDP['norm'], SDP['R'], SDP['z'])
    if splashback_cutoff:  # cut off density at SB radius
        DMP = get_dark_matter_parameters(virial_mass)
        stellar_disc_density[rgrid >= DMP['SB']] = 0
    return stellar_disc_density


def get_densities(virial_mass: float, rgrid: np.ndarray, thgrid: np.ndarray,
                  splashback_cutoff: bool = True, total: bool = False) -> dict:
    """
    Calculates all density components for a given mass, assuming mean 
    values on all empirical relations. Packages the results in a dictionary.

    Parameters:
    virial_mass (float): The virial mass of the halo.
    rgrid (numpy array): The radial grid.
    thgrid (numpy array): The angular grid.
    splashback_cutoff (bool): Whether to cut off the density at 
        the splashback radius.
    total (bool): Whether to include the total density.

    Returns:
    dict: A dictionary of all density components.    
    """
    rho_DM = get_dark_matter_density(
        virial_mass, rgrid, splashback_cutoff)
    rho_SD = get_stellar_disc_density(
        virial_mass, rgrid, thgrid, splashback_cutoff)

    rhos = {'DM': rho_DM,
            'SD': rho_SD}

    if total:
        rhos['total'] = np.zeros_like(rhos['DM'])
        for component, rho in rhos.items():
            if component != 'total':
                rhos['total'] += rhos[component]
    return rhos
