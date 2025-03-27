"""
bias.py

This module contains functions for calculating the halo bias, including a correction
model for the Peak Background Split (PBS) prescription, as described in the provided 
cosmological study. The correction accounts for dependencies on Omega_m(z), the slope 
of the power spectrum, and the clustering amplitude S_8.
"""
import numpy as np

def bias_correction_PBS(Omega_m_z, dlnsigma_dlnR, S8, A0=1.150, a1=0.0929, b1=0.256, b2=0.173, c1=-0.0372):
    """
    Compute the correction factor for the linear halo bias based on the PBS prescription.

    Parameters:
    -----------
    Omega_m_z : float
        The matter density parameter at redshift z.
    dlnsigma_dlnR : float
        The logarithmic derivative of the variance with respect to the radius.
    S8 : float
        The clustering amplitude parameter.

    Returns:
    --------
    float
        The correction factor for the halo bias.
    
    References
    ----------
    - [arxiv:2208.02174](https://arxiv.org/pdf/2311.01465): "Euclid preparation. XXIV. Calibration of the halo mass function in Λ(ν)CDM cosmologies",
      Castro et al., 2023.
    - [arxiv:XXXX.XXXXX](https://arxiv.org/pdf/XXXX.XXXXX): "Euclid preparation. Calibration of the linear halo bias in Λ(ν)CDM cosmologies",
      Castro et al., in prep.
    """
    f0 = 1 + a1 * Omega_m_z
    f1 = 1 + b1 * dlnsigma_dlnR + b2 * dlnsigma_dlnR**2
    f2 = 1 + c1 * S8

    return A0 * f0 * f1 * f2

def corrected_bias(b_PBS, Omega_m_z, dlnsigma_dlnR, S8):
    """
    Calculate the corrected linear halo bias using the PBS prediction and the correction factor.

    Parameters:
    -----------
    b_PBS : float
        The linear halo bias predicted by the PBS model.
    Omega_m_z : float
        The matter density parameter at redshift z.
    dlnsigma_dlnR : float
        The logarithmic derivative of the variance with respect to the radius.
    S8 : float
        The clustering amplitude parameter.

    Returns:
    --------
    float
        The corrected linear halo bias.

    References
    ----------
    - [arxiv:2208.02174](https://arxiv.org/pdf/2311.01465): "Euclid preparation. XXIV. Calibration of the halo mass function in Λ(ν)CDM cosmologies",
      Castro et al., 2023.
    - [arxiv:XXXX.XXXXX](https://arxiv.org/pdf/XXXX.XXXXX): "Euclid preparation. Calibration of the linear halo bias in Λ(ν)CDM cosmologies",
      Castro et al., in prep.
    """
    correction_factor = bias_correction_PBS(Omega_m_z, dlnsigma_dlnR, S8)
    return b_PBS * correction_factor
