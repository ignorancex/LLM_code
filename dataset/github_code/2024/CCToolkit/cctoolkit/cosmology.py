"""
cosmology.py

This module defines the CosmologyCalculator class, which manages cosmological
parameters and calculations related to the power spectrum and other cosmological
quantities using CAMB.
"""
import numpy as np
from camb import CAMBparams, get_results
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .utils import *
from .hmf import *
from .bias import *
from .baryons import *

best_fits = {'AHF': best_fit_values_AHF,
             'ROCKSTAR': best_fit_values_ROCKSTAR,
             'SUBFIND': best_fit_values_SUBFIND,
             'VELOCIraptor': best_fit_values_VELOCIraptor,
             'castro24': best_fit_values_castro24}

class CosmologyCalculator:
    """
    A class to handle cosmological calculations using CAMB.

    This class facilitates the calculation of various cosmological quantities, including
    power spectra, density parameters, and functions related to the Halo Mass Function (HMF).
    It includes methods for calculating the Peak Background Split (PBS) bias and corrected bias,
    utilizing the CAMB library.

    Attributes:
    -----------
    params : dict
        Cosmological parameters for the model, including 'H0', 'Ob0', 'Om0', 'sigma8' or 'As', 'ns', 'TCMB', 'mnu', 'number of massive neutrinos species', 'w0', and 'wa'.
    k : np.ndarray
        Array of wavenumbers used in the power spectrum calculations.
    z_vals : np.ndarray
        Array of redshift values used for power spectrum calculations.
    var : str
        Variable used to compute the matter power spectrum, default is 'cdm+b'.

    Methods:
    --------
    get_power_spectrum(z):
        Returns the wavenumbers and dimensionless power spectrum at a given redshift.
    set_cosmology(new_params):
        Updates the cosmological parameters and recalculates the power spectrum normalization.
    Omega_m(z):
        Returns the matter density parameter Omega_m as a function of redshift.
    Omega_nu(z):
        Returns the neutrino density parameter Omega_nu as a function of redshift.
    Omega_neutrino(z):
        Returns the massless neutrino density parameter Omega_neutrino as a function of redshift.
    Omega_photon(z):
        Returns the photon density parameter Omega_photon as a function of redshift.
    Omega_k(z):
        Returns the curvature density parameter Omega_k as a function of redshift.
    Omega_DE(z):
        Returns the dark energy density parameter Omega_DE as a function of redshift.
    wz(z):
        Returns the dark energy equation of state at a given redshift.
    critical_density(z):
        Returns the critical density at a given redshift.
    H(z):
        Returns the Hubble parameter H at a given redshift.
    peak_height(M, z):
        Calculates the peak-height delta_c / sigma(R(M), z) for a given mass and redshift.
    lagrangian_radius(M):
        Calculates the radius of the Lagrangian patch corresponding to a given mass M.
    sigma(R, z):
        Calculates the RMS fluctuation sigma(R, z) for a given radius and redshift.
    dlnsigma_dlnR(R, z):
        Calculates the derivative of the logarithm of sigma with respect to the logarithm of R.
    vfv(M, z, return_variables=False, halo_finder='ROCKSTAR', model='castro23'):
        Calculates the multiplicity function νf(ν) and related variables for a given mass and redshift.
    dndlnM(M, z, halo_finder='ROCKSTAR', model='castro23'):
        Calculates the halo mass function dn/dlnM, representing the number density of halos per logarithmic mass interval.
    pbs_bias(M, z, halo_finder='ROCKSTAR', hmf_model='castro23', return_variables=False):
        Calculates the PBS bias for halos of mass M at redshift z.
    bias(M, z, halo_finder='ROCKSTAR', hmf_model='castro23'):
        Calculates the corrected linear halo bias for given mass and redshift, using the PBS prediction and correction factors.

    Notes:
    ------
    - This class requires CAMB to be installed and available for the power spectrum calculations.
    - The methods assume that the input masses (M) are in units of solar masses over h (M_sun/h), and distances (R) are in Mpc/h.
    - The halo finder parameters must be provided correctly to ensure accurate mass function and bias calculations.
    """

    def __init__(self, params=None, power_spectrum=None, zmax=2.0, nz=100, var='cdm+b'):
        """
        Initialize the CosmologyCalculator with given cosmological parameters.

        Parameters:
        -----------
        params : dict, optional
            Dictionary containing the cosmological parameters, including 'H0', 'Ob0', 'Om0', 'sigma8',
            'ns', 'TCMB', 'mnu', num_massive_neutrinos, w0, and wa. Default: PICCOLO C0 cosmology.
        power_spectrum : array, optional
            Array containing the power spectrum data (k and Pk). If None, the power spectrum
            will be computed from the parameters using CAMB.
        zmax : float, optional
            Maximum redshift for the calculation. Default is 2.0.
        nz : int, optional
            Number of redshift points. Default is 100.
        var : str, optional
            Which variable to use to compute the matter power spectrum. Default is 'cdm+b' (no neutrinos).
        """

        if params is None:
            params = {}
        self.zmax = zmax
        self.nz = nz
        self._z_vals_inv = np.linspace(0, zmax, nz)[::-1]

        if var == 'cdm+b':
            self._transfer_var = 8
        elif var == 'tot':
            self._transfer_var = 7
        else:
            raise RuntimeError(f"Not supported matter power-spectrum for var={var}!")

        if power_spectrum is not None:
            if 'As' in params:
                raise RuntimeError(f"Only sigma8 is supported (not As) when passing a tabulated P(k)!")
            self._initialize_camb(params, background_only=True)
            self._set_growth_factor()
            self._load_power_spectrum(power_spectrum)
        else:
            self._initialize_camb(params)
            self.growth_factor = None # If we are going to use camb as our backend we do not need growth factor

    def _initialize_camb(self, params, background_only=False):
        """
        Initialize CAMB with the given cosmological parameters.

        Parameters:
        -----------
        params : dict
            Dictionary containing the cosmological parameters.
        background_only: bool
            Whether matter power-spectrum is expected to be computed or not.
        """
        
        # Default parameters
        _H0  = 67.321
        _Ob0 = 0.0494
        _Om0 = 0.3158
        _TCMB = 2.7255
        _mnu  = 0.06
        _num_massive_neutrinos = 1
        _w0 = -1.0
        _wa = 0.0
        _As = 2e-9
        _ns = 0.9661
        _s8 = 0.8102
        if ('As' in params) and ('sigma8' in params):
            raise RuntimeError('Define either `As` or `sigma8`, not both!')
        self._ReNormBool = ('sigma8' in params) or ( ('sigma8' not in params) and ('As' not in params) )

        # inflating parameters if needed
        self.params = {'H0':params.get('H0', _H0),
                       'Ob0':params.get('Ob0', _Ob0),
                       'Om0':params.get('Om0', _Om0),
                       'TCMB':params.get('TCMB', _TCMB),
                       'mnu':params.get('mnu', _mnu),
                       'num_massive_neutrinos':params.get('num_massive_neutrinos', _num_massive_neutrinos),
                       'As':params.get('As', _As),
                       'sigma8':params.get('sigma8', _s8),
                       'w0':params.get('w0', _w0),
                       'wa':params.get('wa', _wa),
                       'ns':params.get('ns', _ns)}
        params = self.params
        
        camb_params = CAMBparams()
        camb_params.set_cosmology(
            H0=params.get('H0', _H0),
            ombh2=params.get('Ob0', _Ob0) * (params.get('H0', _H0) / 100) ** 2,
            omch2=(params.get('Om0', _Om0) - params.get('Ob0', _Ob0)) * (params.get('H0', _H0) / 100) ** 2,
            TCMB=params.get('TCMB', _TCMB),  # Default CMB temperature is 2.7255 K
            mnu=params.get('mnu', _mnu),  # Default sum of neutrino masses in eV
            num_massive_neutrinos=params.get('num_massive_neutrinos', _num_massive_neutrinos)  # Number of massive neutrinos
        )
        camb_params.set_dark_energy(w=params.get('w0', _w0), 
                                    cs2=1.0, 
                                    wa=params.get('wa', _wa), dark_energy_model='ppf')
        if not background_only:
            camb_params.InitPower.set_params(As=params.get('As', _As), ns=params.get('ns', _ns))
            camb_params.WantTransfer = True
            camb_params.set_matter_power(redshifts=self._z_vals_inv, kmax=10.0, k_per_logint=20)
        results = get_results(camb_params)
        self._cosmo = results

        if not background_only:
            self._Pk, self.z_vals, self.k = results.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True, log_interp=False,
                                                                                  var1=self._transfer_var, var2=self._transfer_var, return_z_k=True, extrap_kmax=True)
            self._Dk0 = self._Pk(0, np.log(self.k))[0] * self.k ** 3 / (2*np.pi**2)
            self._Norm = compute_sigma8_norm(self.k, self._Dk0, params.get('sigma8', _s8)) if self._ReNormBool else 1.0
            self.Pk = lambda z: self._Pk(z, np.log(self.k)) / self._Norm
            self.Dk = lambda z: self._Pk(z, np.log(self.k)) / self._Norm * self.k ** 3 / (2*np.pi**2)
            self.params['sigma8'] = self.sigma(8, 0)[0]

    def _load_power_spectrum(self, power_spectrum):
        """
        Load the power spectrum data.

        Parameters:
        -----------
        power_spectrum_file : str
            Path to the file containing the power spectrum data (k and Pk).
        """
        self.k, Pk = power_spectrum
        self._Dk = Pk.flatten() * self.k ** 3 / (2 * np.pi**2)
        self._Dk /= compute_sigma8_norm(self.k, self._Dk, self.params['sigma8'])
        # Reshaping to have the same dimensions as camb
        self._Dk = self._Dk.reshape(1, self._Dk.size)
        self.Dk = lambda z: self._Dk * self.growth_factor(z)**2
    
    def get_power_spectrum(self, z=0):
        """
        Get the wavenumber and dimensionless power spectrum arrays at a given redshift.

        Parameters:
        -----------
        z : float, optional
            Redshift at which to get the power spectrum. Default is 0.

        Returns:
        --------
        tuple
            (k, Dk) where k is the array of wavenumbers and Dk is the dimensionless power spectrum.
        """
        return self.k, self.Dk(z)

    def set_cosmology(self, new_params):
        """
        Update cosmological parameters and recalculate the power spectrum normalization.

        Parameters:
        -----------
        new_params : dict
            Dictionary of new cosmological parameters.
        """
        self.params = new_params
        self._initialize_camb(new_params)

    def Omega_m(self, z):
        """
        Get the matter density parameter Omega_m at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The matter density parameter Omega_m at redshift z.
        """
        return self._cosmo.get_Omega('cdm', z) + self._cosmo.get_Omega('baryon', z)
    
    def Omega_nu(self, z):
        """
        Get the massive neutrino density parameter Omega_nu at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The neutrino density parameter Omega_nu at redshift z.
        """
        return self._cosmo.get_Omega('nu', z)
    
    def Omega_neutrino(self, z):
        """
        Get the massless neutrino density parameter Omega_neutrino at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The massless neutrino density parameter Omega_neutrino at redshift z.
        """
        return self._cosmo.get_Omega('neutrino', z)
    
    def Omega_photon(self, z):
        """
        Get the photon density parameter Omega_photon at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The photon density parameter Omega_neutrino at redshift z.
        """
        return self._cosmo.get_Omega('photon', z)
    
    def Omega_k(self, z):
        """
        Get the curvature density parameter Omega_k at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The curvature density parameter Omega_k at redshift z.
        """
        return self._cosmo.get_Omega('k', z)
    
    def Omega_DE(self, z):
        """
        Get the dark energy density parameter Omega_DE at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The dark energy density parameter Omega_DE at redshift z.

        """
        return self._cosmo.get_Omega('de', z)
    
    def wz(self, z):
        """
        Get the dark energy equation of state at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The dark energy equation of state at redshift z.

        """
        return self._cosmo.get_dark_energy_rho_w(1.0/(1+z))[1]
    
    def zta(self, z):
        """
        Get the turnaround redshift for a given collapse redshift z.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            EdS solution for the turnaroudn redshift.

        """
        return (1+z)/(1/2)**(2/3) - 1

    def critical_density(self, z):
        """
        Get the critical density at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The critical density at redshift z in units of 10^10 M_sun / (h^2 Mpc^3).
        """
        H0 = self.params['H0']  # Hubble parameter today in km/s/Mpc
        H_z = self._cosmo.hubble_parameter(z)  # H(z) in km/s/Mpc
        G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
        Mpc_to_m = 3.085677581491367e22  # Conversion from Mpc to meters
        Msun_to_kg = 1.98847e30  # Solar mass to kg
        h = H0 / 100

        # Convert H(z) to SI units (1/s)
        H_z_si = H_z * 1000 / Mpc_to_m  # km/s/Mpc to 1/s

        # Critical density in SI units (kg/m^3)
        rho_crit_si = (3 * H_z_si**2) / (8 * np.pi * G)

        # Convert to 10^10 M_sun / (h^2 Mpc^3)
        rho_crit = rho_crit_si * Mpc_to_m**3 / Msun_to_kg * 1e-10 / h**2

        return rho_crit
    
    def H(self, z):
        """
        Get the Hubble parameter H at a given redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        Returns:
        --------
        float
            The Hubble parameter H(z) in km/s/Mpc at redshift z.
        """
        return self._cosmo.hubble_parameter(z)
    
    def peak_height(self, M, z):
        """
        Calculate the peak-height delta_c / sigma(R(M), z).

        Parameters:
        -----------
        M : float
            Mass in solar masses over h.
        z : float
            Redshift.

        Returns:
        --------
        float
            The peak-height.
        """
        Om_z = self.Omega_m(z)
        delta_c_z = delta_c(Om_z)
        R = self.lagrangian_radius(M)
        sigma_R_z = self.sigma(R, z)
        return delta_c_z / sigma_R_z
    
    def lagrangian_radius(self, M):
        """
        Calculate the radius of the Lagrangian patch corresponding to a given mass M.

        Parameters:
        -----------
        M : float
            Mass in solar masses over h.

        Returns:
        --------
        float
            The Lagrangian radius in Mpc/h.
        """
        rho_m0 = self.critical_density(0) * self.Omega_m(0) * 1e10  # in M_sun / Mpc^3 * h^2
        R = (3 * M / (4 * np.pi * rho_m0))**(1/3)
        return R
    
    def sigma(self, R, z):
        """
        Calculate the RMS fluctuation sigma(R, z).

        Parameters:
        -----------
        R : float
            Radius in Mpc/h.
        z : float
            Redshift.

        Returns:
        --------
        float
            The RMS fluctuation sigma.
        """
        R = np.atleast_1d(R)
        sigma_squared = ssq_given_Dk(R[:, np.newaxis], self.k, self.Dk(z))
        return np.sqrt(sigma_squared)

    def dlnsigma_dlnR(self, R, z):
        """
        Calculate the derivative of the logarithm of sigma with respect to the logarithm of R.

        Parameters:
        -----------
        R : float
            Radius in Mpc/h.
        z : float
            Redshift.

        Returns:
        --------
        float
            The derivative dln(sigma)/dln(R).
        """
        R = np.atleast_1d(R)
        d_sigma_dR = d_s_given_Dk(R[:, np.newaxis], self.k, self.Dk(z))
        sigma_R = self.sigma(R, z)

        # Avoiding newaxis to create a matrix when multiplying 
        return (R.flatten() / sigma_R) * d_sigma_dR
    
    def vfv(self, M, z, return_variables=False, halo_finder='ROCKSTAR', model='castro23'):
        """
        Calculate the multiplicity function, defined as νf(ν), where ν is the peak-height, as well as 
        related variables for a given mass and redshift.

        Parameters:
        -----------
        M : np.ndarray
            Mass of the halos in solar masses over h. 
        z : float
            Redshift. 
        return_variables : bool, optional
            If True, returns additional intermediate variables used in the calculation.
            If False, returns only the multiplicity function. Default is False.
        halo_finder : str, optional
            Only used for model `castro23`. Descriptor for the best-fit values for the mass function parameters.
            Default is `ROCKSTAR`. Options are: `AHF`, `ROCKSTAR`, `SUBFIND`, and `VELOCIraptor`.
        model : str, optional
            Descriptor for the multiplicity function model.
            Default is `castro23`. Options are `castro23` and `castro24`.

        Returns:
        --------
        np.ndarray or tuple
            If `return_variables` is False, returns an array of the multiplicity function νf(ν).
            If `return_variables` is True, returns a tuple containing:
            - M : Mass array
            - R : Lagrangian radius array corresponding to M
            - ν : Peak-height array
            - dlnσ/dlnR : Derivative of log sigma with respect to log R
            - νf(ν) : Multiplicity function values

        Raises:
        -------
        ValueError
            If an unsupported model is specified.

        Notes:
        ------
        For the 'castro24' model, the `halo_finder` parameter is ignored, and appropriate best-fit values are used.
        """
        # Validate inputs
        if not isinstance(M, np.ndarray):
            raise ValueError("Masses should be an instance of numpy.ndarray")
        if model not in ['castro23', 'castro24']:
            raise ValueError(f"Model '{model}' is not implemented.")
        
        # Compute variables
        R = self.lagrangian_radius(M)
        v = self.peak_height(M, z)
        dlnsdlnR = self.dlnsigma_dlnR(R, z)

        # Access best-fit parameters
        if model == 'castro23':
            # Ensure halo_finder is valid
            valid_halo_finders = ['AHF', 'ROCKSTAR', 'SUBFIND', 'VELOCIraptor']
            if halo_finder not in valid_halo_finders:
                raise ValueError(f"Invalid halo_finder '{halo_finder}'. Valid options are {valid_halo_finders}.")
            best_fit_values = best_fits[halo_finder]
        elif model == 'castro24':
            best_fit_values = best_fits['castro24']
        else:
            raise ValueError(f"Model '{model}' is not implemented.")

        # Prepare additional parameters for castro24
        if model == 'castro24':
            zta = self.zta(z)
            Omega_de_zta = self.Omega_DE(zta)
            w_de_zta = self.wz(zta)
            extra_params = {
                'Omega_de_zta': Omega_de_zta,
                'w_de_zta': w_de_zta
            }
        else:
            extra_params = {
                'Omega_de_zta': None,
                'w_de_zta': None
            }

        # Compute multiplicity function
        f_nu = multiplicity_function(
            peak_height=v,
            dlns_dlnR=dlnsdlnR,
            Omega_m_z=self.Omega_m(z),
            best_fit_values=best_fit_values,
            model=model,
            **extra_params
        )

        if return_variables:
            return M, R, v, dlnsdlnR, f_nu
        else:
            return f_nu

    def dndlnM(self, M, z, halo_finder='ROCKSTAR', model='castro23'):
        """
        Calculate the halo mass function, dn/dlnM, which gives the number density of halos 
        per logarithmic mass interval.

        Parameters:
        -----------
        M : np.ndarray
            Mass of the halos in solar masses over h. 
        z : float
            Redshift. 
        halo_finder : str, optional
            Only used for model `castro23`. Descriptor for the best-fit values for the mass function parameters.
            Default is `ROCKSTAR`. Options are: `AHF`, `ROCKSTAR`, `SUBFIND`, and `VELOCIraptor`.
        model : str, optional
            Descriptor for the multiplicity function model.
            Default is `castro23`. Options are `castro23` and `castro24`.

        Returns:
        --------
        np.ndarray
            The halo mass function dn/dlnM, with units of number density per logarithmic mass interval.

        Raises:
        -------
        ValueError
            If an unsupported model is specified.

        Notes:
        ------
        For the 'castro24' model, the `halo_finder` parameter is ignored.
        """
        # Validate inputs
        if not isinstance(M, np.ndarray):
            raise ValueError("Masses should be an instance of numpy.ndarray")
        if model not in ['castro23', 'castro24']:
            raise ValueError(f"Model '{model}' is not implemented.")

        # Compute vfv and related variables
        M, R, v, dlnsdlnR, vfv_values = self.vfv(
            M, z, halo_finder=halo_finder, model=model, return_variables=True
        )

        # Calculate dn/dlnM
        dn_dlnM = (
            self.critical_density(0.0) * self.Omega_m(0.0) * 1e10 / M * vfv_values * (-1 / 3 * dlnsdlnR)
        )

        return dn_dlnM
    
    def pbs_bias(self, M, z, halo_finder='ROCKSTAR', return_variables=False, hmf_model='castro23'):
        """
        Calculate the PBS bias for halos of mass M at redshift z.

        Parameters:
        -----------
        M : np.ndarray
            Mass of the halos in solar masses over h. 
        z : float
            Redshift.
        halo_finder : str
            The halo finder used to determine the best-fit parameters. Default is 'ROCKSTAR'.
        return_variables : bool, optional
            If True, returns additional intermediate variables used in the calculation.
            If False, returns only the bias. Default is False.
        hmf_model : str
            The hmf model used for the multiplicity function. Default is 'castro23'.

        Returns:
        --------
        np.ndarray or tuple
            If `return_variables` is False, returns an array of the PBS bias.
            If `return_variables` is True, returns a tuple containing:
            - M : Mass array
            - R : Lagrangian radius array corresponding to M
            - v : Peak-height array
            - dlnsdlnR : Derivative of log sigma with respect to log R
            - vfv : Multiplicity function values
            - bias : PBS bias values
        """
        if M.size < 2:
            raise ValueError("Mass array should contain at least 2 elements.")
        # Since we need the edges to be accurate in their derivative, let's extend the mass array
        M = np.insert(M, [0, M.size], [0.95*M.min(), 1.05*M.max()])
        M, R, v, dlnsdlnR, vfv = self.vfv(M, z, halo_finder=halo_finder, return_variables=True, model=hmf_model)
        _delta_c = delta_c(self.Omega_m(z))
        
        # Avoid division by zero or log of zero issues
        if np.any(v <= 0) or np.any(vfv <= 0):
            raise ValueError("Invalid values in v or vfv arrays; check input mass or redshift ranges.")
        
        bias = 1 - 1/_delta_c * np.gradient(np.log(vfv), np.log(v))
        if return_variables:
            return M[1:-1], R[1:-1], v[1:-1], dlnsdlnR[1:-1], vfv[1:-1], bias[1:-1]
        else:
            return bias[1:-1]

    def bias(self, M, z, halo_finder='ROCKSTAR', hmf_model='castro23'):
        """
        Calculate the corrected linear halo bias.

        Parameters:
        -----------
        M : np.ndarray
            Mass of the halos in solar masses over h. Can be a scalar or array.
        z : float
            Redshift.
        halo_finder : str
            The halo finder used to determine the best-fit parameters. Default is 'ROCKSTAR'.
        hmf_model : str
            The hmf model used for the multiplicity function. Default is 'castro23'.

        Returns:
        --------
        np.ndarray
            The corrected linear halo bias.
        """
        M, R, v, dlnsdlnR, vfv, b_pbs = self.pbs_bias(M, z, halo_finder=halo_finder, return_variables=True, hmf_model=hmf_model)
        Omz = self.Omega_m(z)
        S8 = self.sigma(8, 0.0) * np.sqrt(self.Omega_m(0.0)/0.3)
    
        return corrected_bias(b_pbs, Omz, dlnsdlnR, S8)

    def _set_growth_factor(self):
        """
        Solve for the growth factor with appropriate initial conditions.

        The growth factor is computed based on the effective treatment described by Linder et al 2003. The method sets 
        different initial conditions depending on the value of the cosmic microwave background temperature (TCMB). The 
        differential equation is solved using the `solve_ivp` method from `scipy.integrate`.

        The method sets the `growth_factor` attribute, which is an interpolated function of the growth factor normalized 
        to the present day.

        Notes
        -----
        - If `TCMB` is very low, Einstein-de Sitter (EdS) conditions are used.
        - Otherwise, radiation-dominated era conditions are more appropriate.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Attributes
        ----------
        growth_factor : scipy.interpolate.interp1d
            An interpolated function of the normalized growth factor.

        References
        ----------
        E. V. Linder, A. Jenkins, Cosmic structure growth and dark energy, Monthly Notices of the Royal Astronomical Society, Volume 346, Issue 2, December 2003, Pages 573–583, https://doi.org/10.1046/j.1365-2966.2003.07112.x
        """
        
        # Checking which conditions to use. If TCMB is very low, EdS conditions are better
        if self.params['TCMB'] < 1:
            a_init    = 1e-4
            a_end     = 1
            g_init    = a_init
            dgda_init = 0
        # Otherwise a radiation dominated era is more convenient
        else:
            a_init  = 1e-7
            a_end   = 1
            g_init    = np.log(a_init)/a_init
            dgda_init = (1-np.log(a_init)) / a_init**2
        y0 = [g_init, dgda_init]
        # Defining the effective functions (Linder et al 2003)
        a_vals  = np.geomspace(a_init, a_end, 1000)
        z_vals  = 1/a_vals-1
        deltaH2 = ( self.H(z_vals)/self.H(0) )**2 - self.Omega_m(0) / a_vals**3
        weff    = -1 - 1/3 * np.gradient(np.log(deltaH2), np.log(a_vals))
        Xeff    = self.Omega_m(0) / a_vals**3 / deltaH2

        def growth_system(a, y):
            g, dgda = y
            wa = np.interp(a, a_vals, weff)
            Xa = np.interp(a, a_vals, Xeff)

            d2gda2 = - (7/2 - 3/2*wa/(1+Xa)) * dgda/a - 3/2* (1 - wa)/(1 + Xa) * g / a**2

            return [dgda, d2gda2]
        
        # Solve the differential equation using solve_ivp
        sol = solve_ivp(growth_system, (a_init, a_end), y0, dense_output=True)
        # Extract the solution for g
        g = sol.y[0]
        g = g/g[-1]
        a = sol.t[::-1]
        z = 1/a - 1
        g = g[::-1]

        self.growth_factor = interp1d(z[z<200], (g*a)[z<200], kind='quadratic')
