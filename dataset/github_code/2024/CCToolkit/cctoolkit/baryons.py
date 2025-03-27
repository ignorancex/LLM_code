"""
baryons.py

This module implements the model for the impact of baryonic effects on the masses of clusters
and groups presented in Castro et al. 2024 (https://inspirehep.net/literature/2718844).
"""
import numpy as np
from scipy.optimize import fsolve
from .utils import critical_density0, virial_Delta

def baryon_fraction_magneticum(M_vir, z):
    """
    Mean baryon fraction contained inside the virial radius for Magneticum simulations.

    Parameters
    ----------
    M_vir : float
        Virial mass.
    z : float
        Redshift.

    Returns
    -------
    fb_fid : float
        The mean baryon fraction inside the virial radius for Magneticum clusters.
    """
    fb_cosmic = 0.168  # Magneticum baryon fraction
    m = np.log10(M_vir / 1e10) / 4
    gamma = 0.7008 * z - 1.7505
    delta = 0.2
    fb_mean = fb_cosmic * (m ** -gamma) * (1 + m ** (1 / delta)) ** (gamma * delta)
    return fb_mean

def baryon_fraction_tng300(M_vir, z):
    """
    Mean baryon fraction contained inside the virial R for TNG300 simulations.

    Parameters
    ----------
    M_vir : float
        Virial mass.
    z : float
        Redshift.

    Returns
    -------
    fb_fid : float
        The mean baryon fraction inside the virial radius for TNG300 clusters.
    """
    fb_cosmic = 0.1573  # TNG cosmic baryon fraction
    m = np.log10(M_vir / 1e10) / 4
    gamma = 0.4106 * z - 1.4640
    delta = 0.0397 * z ** 2 - 0.04358 * z + 0.03567
    fb_mean = fb_cosmic * (m ** -gamma) * (1 + m ** (1 / delta)) ** (gamma * delta)
    return fb_mean

def compute_dmo_mass(M_vir_hydro, z, fb_cosmic, relation='magneticum'):
    """
    Compute the SO equivalent DMO mass and overdensity Delta (virial units) to
    the virial object definition from hydrodynamic simulations.

    Parameters
    ----------
    M_vir_hydro : float
        Virial mass from hydrodynamic simulations.
    z : float
        Redshift.
    fb_cosmic : float
        Cosmic baryon fraction.
    relation : str or function, optional
        The baryon fraction relation to use, either 'magneticum' or 'tng300' or a custom function with signature (M, z).
        Default is 'magneticum'.

    Returns
    -------
    M_delta_dmo : float
        The equivalent DMO mass at the Delta overdensity.
    Delta : float
        The overdensity of equivalence in virial units.

    Raises
    ------
    ValueError
        If an invalid relation is specified.
    """
    if isinstance(relation, str):
        if relation == 'magneticum':
            fb_hydro_vir = baryon_fraction_magneticum(M_vir_hydro, z)
        elif relation == 'tng300':
            fb_hydro_vir = baryon_fraction_tng300(M_vir_hydro, z)
        else:
            raise ValueError("Invalid relation specified. Use 'magneticum', 'tng300' or a custom function with identical signature.")
    elif callable(relation):
        fb_hydro_vir = relation(M_vir_hydro, z)
    else:
        raise ValueError("Invalid relation specified. Use 'magneticum', 'tng300' or a custom function with identical signature.")

    delta_f = (0.045 - 0.005 * z) * fb_cosmic  # Baryonic offset parameter
    q = 0.373  # Quasi-adiabatic parameter

    # Step 1: Calculate M∆,dmo using Eq. (5)
    x = (1 - fb_hydro_vir - delta_f) / (1 - fb_cosmic)
    M_Delta_dmo = M_vir_hydro * x

    # Step 3: Calculate the DMO spherical overdensity threshold ∆ using Eq. (4) with respect to the virial Delta
    Delta_over_Delta_vir = x / (1 + q * (1 / x - 1)) ** 3

    return M_Delta_dmo, Delta_over_Delta_vir

def concentration_duffy_et_al_2008(M, z):
    """
    Calculate the concentration-mass relation according to Duffy et al. (2008) for virial masses.

    This function implements the formula provided by Duffy et al. (2008) for determining
    the concentration parameter as a function of mass and redshift. The formula is given by:

    .. math::

        c(M, z) = A_{vir} \\left(\\frac{M}{M_{piv}}\\right)^{B_{vir}} (1+z)^{C_{vir}}

    where the parameters are defined as follows:
    - :math:`A_{vir} = 7.85`
    - :math:`B_{vir} = -0.081`
    - :math:`C_{vir} = -0.71`
    - :math:`M_{piv} = 2 \\times 10^{12} M_\odot`

    Parameters
    ----------
    M : float
        Virial mass of the halo in solar masses.
    z : float
        Redshift at which the concentration is calculated.

    Returns
    -------
    float
        Concentration parameter for the given mass and redshift.

    References
    ----------
    Duffy, A. R., et al. (2008). "Dark matter halo concentrations in the Wilkinson
    Microwave Anisotropy Probe year 5 cosmology." Monthly Notices of the Royal
    Astronomical Society, 390(2), L64-L68.
    """
    Avir = 7.85
    Bvir = -0.081
    Cvir = -0.71
    Mpv  = 2e12

    return Avir * (M/Mpv) ** Bvir * (1+z)**Cvir


def compute_rec_mass(cosmo, M_Delta_dmo, Delta, z, c=concentration_duffy_et_al_2008):
    """
    Reconstruct the dark matter-only (DMO) mass of a hydro object using the model
    presented in [arxiv:2311.01465](https://arxiv.org/pdf/2311.01465).

    This function calculates the reconstructed mass of a hydro object, denoted as
    \( M_{\Delta,\text{dmo}} \), using the provided concentration-mass relation
    and cosmological parameters.

    Parameters
    ----------
    cosmo : object
        Cosmology object containing cosmological parameters and methods to compute
        relevant quantities, such as the critical density and Omega_m(z).
    M_Delta_dmo : float
        Mass of the dark matter-only object within radius defined by \(\Delta\) times
        the critical density, in solar masses.
    Delta : float
        Overdensity parameter, defining the radius within which the mass
        \( M_{\Delta,\text{dmo}} \) is calculated.
    z : float
        Redshift at which the calculation is performed.
    c : function, optional
        Function to compute the concentration parameter given mass and redshift.
        Default is `concentration_duffy_et_al_2008`.

    Returns
    -------
    float
        Reconstructed virial mass of the object at redshift `z`.

    Notes
    -----
    The function first computes the virial overdensity parameter, then iteratively
    solves for the virial radius \( R_{\text{vir}} \) using the auxiliary NFW
    function. The virial mass is finally obtained using the computed radius and
    density.

    References
    ----------
    - [arxiv:2311.01465](https://arxiv.org/pdf/2311.01465): "Euclid preparation XXXIX. The effect of baryons on the halo mass function",
      Castro et al., 2024.

    """
    R_Delta_dmo = np.cbrt(M_Delta_dmo / (4*np.pi*critical_density0*Delta/3))
    Delta_vir = virial_Delta(cosmo.Omega_m(z))

    # Auxiliary NFW function
    f = lambda x, _c: np.log(1+_c*x) - _c*x/(1+_c*x)
    # Auxiliary function to be solved
    def fRvir (R):

        Mvir = 4 * np.pi/3 * R**3 * critical_density0 * Delta_vir
        cvir = c(Mvir, z)
        c_Delta = R_Delta_dmo/R * cvir

        return M_Delta_dmo * f(R/R_Delta_dmo, c_Delta) / f(1, c_Delta) / (4*np.pi/3*critical_density0*R**3) - Delta_vir

    Rvir = fsolve(fRvir, 0.99 * R_Delta_dmo)[0]

    return (4*np.pi/3) * Rvir**3 * critical_density0 * Delta_vir
