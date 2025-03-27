"""
hmf.py

This module contains functions and data related to the Halo Mass Function (HMF),
including the multiplicity function and best-fit parameters for different halo finders.
"""

import numpy as np
from scipy.special import gamma

def multiplicity_function(peak_height, dlns_dlnR, Omega_m_z, best_fit_values, model="castro23", Omega_de_zta=None, w_de_zta=None):
    """
    Handler for the multiplicity function.

    Parameters:
    -----------
    peak_height : float
        The peak height (ν).
    dlns_dlnR : float
        The logarithmic slope of the variance with respect to the radius.
    Omega_m_z : float
        The matter density parameter at redshift z.
    best_fit_values : dict
        Dictionary containing the best-fit parameters for the chosen halo finder.
    model : str, optional
        The model to use for the multiplicity function. Options are 'castro23' (default) and 'castro24'.
    Omega_de_zta : float, optional
        The dark energy density parameter at turn-around. Required if model is 'castro24'.
    w_de_zta : float, optional
        The dark energy equation of state parameter at turn-around. Required if model is 'castro24'.

    Returns:
    --------
    float
        The value of the multiplicity function.

    Raises:
    -------
    ValueError
        If required parameters are not provided for the selected model.

    References
    ----------
    - [arXiv:2208.02174](https://arxiv.org/pdf/2208.02174): "Euclid preparation. XXIV. Calibration of the halo mass function in Λ(ν)CDM cosmologies",
      Castro et al., 2023.
    - [arXiv:xxxx.xxxx](https://arxiv.org/pdf/xxxx.xxxx): "DUCA: Dynamic Universe Cosmological Analysis. The halo mass function in dynamic dark energy cosmologies",
      Castro et al., 2024.
    """
    if model == 'castro23':
        return multiplicity_function_castro23(peak_height, dlns_dlnR, Omega_m_z, best_fit_values)
    elif model == 'castro24':
        if Omega_de_zta is None or w_de_zta is None:
            raise ValueError("Omega_de_zta and w_de_zta must be provided for model 'castro24'.")
        return multiplicity_function_castro24(peak_height, dlns_dlnR, Omega_m_z, Omega_de_zta, w_de_zta, best_fit_values)
    else:
        raise ValueError(f"Model '{model}' is not implemented.")

def multiplicity_function_castro23(peak_height, dlns_dlnR, Omega_m_z, best_fit_values):
    """
    Calculate the multiplicity function using the Castro et al. 2023 model.

    Parameters:
    -----------
    peak_height : float
        The peak height (ν).
    dlns_dlnR : float
        The logarithmic slope of the variance with respect to the radius.
    Omega_m_z : float
        The matter density parameter at redshift z.
    best_fit_values : dict
        Dictionary containing the best-fit parameters for the chosen halo finder.

    Returns:
    --------
    float
        The value of the multiplicity function.

    References
    ----------
    - [arXiv:2208.02174](https://arxiv.org/pdf/2208.02174): "Euclid preparation. XXIV. Calibration of the halo mass function in Λ(ν)CDM cosmologies",
      Castro et al., 2023.
    """
    # Extract best-fit parameters
    a1 = best_fit_values['a1']
    a2 = best_fit_values['a2']
    az = best_fit_values['az']
    p1 = best_fit_values['p1']
    p2 = best_fit_values['p2']
    q1 = best_fit_values['q1']
    q2 = best_fit_values['q2']
    qz = best_fit_values['qz']

    # Compute coefficients aR, qR, a, p, q
    aR = a1 + a2 * (dlns_dlnR + 0.6125) ** 2
    qR = q1 + q2 * (dlns_dlnR + 0.5)
    a = aR * Omega_m_z ** az
    p = p1 + p2 * (dlns_dlnR + 0.5)
    q = qR * Omega_m_z ** qz

    # Amplitude A(p, q)
    A_pq = (2 ** (-1 / 2 - p + q / 2) / np.sqrt(np.pi) *
            (2 ** p * gamma(q / 2) + gamma(-p + q / 2))) ** (-1)

    # Calculate multiplicity function
    nu = peak_height
    multiplicity = (A_pq * np.sqrt(2 * a * nu ** 2 / np.pi) *
                    np.exp(-a * nu ** 2 / 2) *
                    (1 + 1 / (a * nu ** 2) ** p) *
                    (nu * np.sqrt(a)) ** (q - 1))

    return multiplicity

def multiplicity_function_castro24(peak_height, dlns_dlnR, Omega_m_z, Omega_de_zta, w_de_zta, best_fit_values):
    """
    Calculate the multiplicity function using the Castro et al. 2024 model,
    which accounts for dynamic dark energy cosmologies.

    Parameters:
    -----------
    peak_height : float
        The peak height (ν).
    dlns_dlnR : float
        The logarithmic slope of the variance with respect to the radius.
    Omega_m_z : float
        The matter density parameter at redshift z.
    Omega_de_zta : float
        The dark energy density parameter at turn-around.
    w_de_zta : float
        The dark energy equation of state parameter at turn-around.
    best_fit_values : dict
        Dictionary containing the best-fit parameters for the chosen halo finder.

    Returns:
    --------
    float
        The value of the multiplicity function.

    References
    ----------
    - [arXiv:xxxx.xxxx](https://arxiv.org/pdf/xxxx.xxxx): "DUCA: Dynamic Universe Cosmological Analysis. I. The halo mass function in dynamic dark energy cosmologies",
      Castro et al., 2024.
    """

    # Extract best-fit parameters
    a1 = best_fit_values['a1']
    a2 = best_fit_values['a2']
    az = best_fit_values['az']
    alphaa = best_fit_values['alphaa']
    p1 = best_fit_values['p1']
    p2 = best_fit_values['p2']
    q1 = best_fit_values['q1']
    qz = best_fit_values['qz']

    # Compute fixed coefficients
    q2 = 1.0020 * p2 + 0.7363
    r1 = -0.9917 * q1 + 0.5560
    r2 = -1.1159 * q2 - 0.1189
    vp = 1.0667 * q1 + 1.7577
    alphap = 4.7686 * alphaa + 0.0051

    # Compute coefficients aR, qR, a, p, q, r
    aR = a1 + a2 * (dlns_dlnR + 0.6125) ** 2
    qR = q1 + q2 * (dlns_dlnR + 0.5)
    a = aR * Omega_m_z ** az
    p = p1 + p2 * (dlns_dlnR + 0.5)
    q = qR * Omega_m_z ** qz
    r = r1 + r2 * (dlns_dlnR + 0.5)

    # Apply the DE cosmological modifications
    qdec = 1.5 * (w_de_zta + 1) * Omega_de_zta
    a *= 1 + alphaa * qdec
    p *= 1 + alphap * qdec

    # Amplitude A(p, q, r)
    numerator = (2 ** (-p + q / 2 + 0.5) *
                 (-2 ** (p + r / 2) * q * gamma(q / 2 + r / 2 + 1) +
                  2 ** (p + r / 2 + 1) * p * gamma(q / 2 + r / 2 + 1) -
                  q * vp ** (2 * p) * gamma(-p + q / 2 + 1) -
                  r * vp ** (2 * p) * gamma(-p + q / 2 + 1)))
    denominator = (np.sqrt(np.pi) * (2 * p - q) * (q + r))
    A_pqr = (numerator / denominator) ** (-1)

    # Calculate multiplicity function
    vst = peak_height * np.sqrt(a)
    multiplicity = (2.0 * A_pqr *
                    (vst ** r + (vp / vst) ** (2 * p)) *
                    np.sqrt(vst ** 2 / (2 * np.pi)) *
                    np.exp(-vst ** 2 / 2) *
                    vst ** (q - 1))

    return multiplicity

# Best-fit parameters for different halo finders (Castro et al. 2023)
best_fit_values_ROCKSTAR = {
    'a1': 0.7962, 'a2': 0.1449, 'az': -0.0658,
    'p1': -0.5612, 'p2': -0.4743,
    'q1': 0.3688, 'q2': -0.2804, 'qz': 0.0251
}

best_fit_values_AHF = {
    'a1': 0.7937, 'a2': 0.1119, 'az': -0.0693,
    'p1': -0.5689, 'p2': -0.4522,
    'q1': 0.3652, 'q2': -0.2628, 'qz': 0.0376
}

best_fit_values_SUBFIND = {
    'a1': 0.7953, 'a2': 0.1667, 'az': -0.0642,
    'p1': -0.6265, 'p2': -0.4907,
    'q1': 0.3215, 'q2': -0.2993, 'qz': 0.0330
}

best_fit_values_VELOCIraptor = {
    'a1': 0.7987, 'a2': 0.1227, 'az': -0.0523,
    'p1': -0.5912, 'p2': -0.4088,
    'q1': 0.3634, 'q2': -0.2732, 'qz': 0.0715
}

best_fit_values_castro24 = {
    "a1": 0.8501, "a2": 0.234, "az": -0.0640, "alphaa": -0.178,
    "p1": -0.9880, "p2": -0.49,
    "q1": 0.559, "qz": 0.02701
}