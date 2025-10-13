light_speed = 299792.458  # speed of light in km/s

import numpy as np
from astropy.modeling import models, fitting
from scipy.optimize import brute
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.special import voigt_profile
from scipy.integrate import quad


def integrate_gaussian(a, b, mu, fwhm):
    """Integrate Gaussian between a and b. The Gaussian is not normalized and has central amplitude of 1.

    :param a: lower limit of integration
    :type a: float
    :param b: upper limit of integration
    :type b: float
    :param mu: mean of the Gaussian
    :type mu: float
    :param fwhm: full width at half maximum of the Gaussian
    :type fwhm: float
    :return: integral of the Gaussian between a and b
    :rtype: float
    """
    sigma = fwhm / 2.355
    return (
        sigma
        * np.sqrt(np.pi / 2.0)
        * (erf((b - mu) / (np.sqrt(2) * sigma)) - erf((a - mu) / (np.sqrt(2) * sigma)))
    )


def test_integrate_gaussian():
    """Test the integrate_gaussian function.

    :return: None
    :rtype: None
    """
    assert np.isclose(
        integrate_gaussian(-1, 1, 0, 2.355), 0.6826894921370859 * np.sqrt(2 * np.pi)
    ), "Error in integrate_gaussian function"


test_integrate_gaussian()


def integrate_voigt(a, b, mu, fwhm, gamma):
    """Integrate Voigt profile between a and b. The Voigt profile is not normalized and has central amplitude of 1.

    :param a: lower limit of integration
    :type a: float
    :param b: upper limit of integration
    :type b: float
    :param mu: mean of the Voigt profile
    :type mu: float
    :param fwhm: full width at half maximum of the Voigt profile
    :type fwhm: float
    :param gamma: Lorentzian width of the Voigt profile
    :type gamma: float
    :return: integral of the Voigt profile between a and b
    :rtype: float
    """
    # f_l = 2 * gamma

    # f_s = np.sqrt(
    #     max((fwhm - 0.5343 * f_l) ** 2 - 0.2169 * f_l**2, (fwhm - 0.5343 * f_l) ** 2)
    # )

    sigma = fwhm / 2.355
    integrals = np.zeros_like(a)
    for i, (a_, b_) in enumerate(zip(a, b)):
        integrals[i] = quad(
            voigt_profile,
            a_ - mu,
            b_ - mu,
            args=(sigma, gamma),
            epsrel=1e-6,
            epsabs=1e-6,
        )[0]

    return integrals


def integrate_lorentzian(a, b, mu, fwhm):
    """Integrate Lorentzian profile between a and b. The Lorentzian profile is not normalized and has central amplitude of 1.

    :param a: lower limit of integration
    :type a: float
    :param b: upper limit of integration
    :type b: float
    :param mu: mean of the Lorentzian
    :type mu: float
    :param fwhm: full width at half maximum of the Lorentzian
    :type fwhm: float
    :return: integral of the Lorentzian profile between a and b
    :rtype: float
    """
    gamma = fwhm / 2.0
    return (np.arctan((b - mu) / gamma) - np.arctan((a - mu) / gamma)) / np.pi


def get_lorentzian(x, mu, fwhm, amp=1, continuum_amp=0, continuum_slope=0):
    """Get a Lorentzian profile.

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Lorentzian
    :type mu: float
    :param fwhm: full width at half maximum of the Lorentzian
    :type fwhm: float
    :param amp: amplitude of the Lorentzian
    :type amp: float
    :return: Lorentzian profile
    :rtype: np.ndarray
    """
    gamma = fwhm / 2
    return (
        amp * (gamma) / ((x - mu) ** 2 + (gamma) ** 2) / np.pi
        + continuum_amp
        + (x - np.mean(x)) * continuum_slope
    )


def get_gaussian(
    x,
    mu,
    fwhm,
    gaussian_amp=1,
    continuum_amp=0,
    continuum_slope=0,
):
    """Model function for a non-normalized Gaussian (i.e., central amplitude = 1).

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Gaussian
    :type mu: float
    :param fwhm: full width at half maximum of the Gaussian
    :type fwhm: float
    :param gaussian_amp: amplitude of the Gaussian
    :type gaussian_amp: float
    :param continuum_amp: amplitude of the continuum
    :type continuum_amp: float
    :param continuum_slope: slope of the continuum
    :type continuum_slope: float
    :return: Gaussian model
    :rtype: np.ndarray
    """
    sigma = fwhm / 2.355
    gaussian = (
        gaussian_amp
        # / (x[1] - x[0])  # This is the pixel width
        # / np.sqrt(2 * np.pi)
        # / sigma
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )

    return gaussian + continuum_amp + (x - np.mean(x)) * continuum_slope


def get_voigt(
    x,
    mu,
    fwhm,
    gamma,
    amp=1,
    continuum_amp=0,
    continuum_slope=0,
):
    """Model function for a non-normalized Voigt profile (i.e., central amplitude = 1).

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Voigt profile
    :type mu: float
    :param fwhm: full width at half maximum of the Voigt profile
    :type fwhm: float
    :param gamma: Lorentzian width of the Voigt profile
    :type gamma: float
    :param amp: amplitude of the Voigt profile
    :type amp: float
    :param continuum_amp: amplitude of the continuum
    :type continuum_amp: float
    :param continuum_slope: slope of the continuum
    :type continuum_slope: float
    :return: Voigt model
    :rtype: np.ndarray
    """
    sigma = fwhm / 2.355

    voigt = amp * voigt_profile(x - mu, sigma, gamma)

    return voigt + continuum_amp + (x - np.mean(x)) * continuum_slope


def get_pixel_integrated_gaussian(
    x,
    mu,
    fwhm,
    gaussian_amp=1,
    continuum_amp=0,
    continuum_slope=0,
):
    """Model function for pixel-integrated Gaussian

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Gaussian
    :type mu: float
    :param fwhm: full width at half maximum of the Gaussian
    :type fwhm: float
    :param gaussian_amp: amplitude of the Gaussian
    :type gaussian_amp: float
    :param continuum_amp: amplitude of the continuum
    :type continuum_amp: float
    :param continuum_slope: slope of the continuum
    :type continuum_slope: float
    :return: Gaussian model
    :rtype: np.ndarray
    """
    lambda_diff = x[1] - x[0]

    integrated_gaussian = gaussian_amp * integrate_gaussian(
        x - lambda_diff / 2.0, x + lambda_diff / 2.0, mu, fwhm
    )
    return integrated_gaussian + continuum_amp + (x - np.mean(x)) * continuum_slope


def get_pixel_integrated_voigt(
    x,
    mu,
    fwhm,
    gamma,
    amp=1,
    continuum_amp=0,
    continuum_slope=0,
):
    """Model function for pixel-integrated Voigt profile.

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Voigt profile
    :type mu: float
    :param fwhm: full width at half maximum of the Voigt profile
    :type fwhm: float
    :param gamma: Lorentzian width of the Voigt profile
    :type gamma: float
    :param amp: amplitude of the Voigt profile
    :type amp: float
    :param continuum_amp: amplitude of the continuum
    :type continuum_amp: float
    :param continuum_slope: slope of the continuum
    :type continuum_slope: float
    :return: Voigt model
    :rtype: np.ndarray
    """
    lambda_diff = x[1] - x[0]

    integrated_voigt = amp * integrate_voigt(
        x - lambda_diff / 2.0, x + lambda_diff / 2.0, mu, fwhm, gamma
    )

    return integrated_voigt + continuum_amp + (x - np.mean(x)) * continuum_slope


def get_pixel_integrated_lorentzian(
    x,
    mu,
    fwhm,
    amp=1,
    continuum_amp=0,
    continuum_slope=0,
):
    """Model function for pixel-integrated Lorentzian profile.

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Lorentzian profile
    :type mu: float
    :param fwhm: full width at half maximum of the Lorentzian profile
    :type fwhm: float
    :param amp: amplitude of the Lorentzian profile
    :type amp: float
    :param continuum_amp: amplitude of the continuum
    :type continuum_amp: float
    :param continuum_slope: slope of the continuum
    :type continuum_slope: float
    :return: Lorentzian model
    :rtype: np.ndarray
    """
    lambda_diff = x[1] - x[0]

    integrated_lorentzian = amp * integrate_lorentzian(
        x - lambda_diff / 2.0, x + lambda_diff / 2.0, mu, fwhm
    )

    return integrated_lorentzian + continuum_amp + (x - np.mean(x)) * continuum_slope


def get_spectra_model(
    x,
    mu,
    fwhm,
    gaussian_amp=1,
    continuum_amp=0,
    continuum_slope=0,
    line_type="gaussian",
    voigt_gamma=1.0,
):
    """Model function for a Gaussian for spectra.

    :param x: wavelengths
    :type x: np.ndarray
    :param mu: mean of the Gaussian
    :type mu: float
    :param fwhm: full width at half maximum of the Gaussian
    :type fwhm: float
    :param gaussian_amp: amplitude of the Gaussian
    :type gaussian_amp: float
    :param continuum_amp: amplitude of the continuum
    :type continuum_amp: float
    :param continuum_slope: slope of the continuum
    :type continuum_slope: float
    :return: Gaussian model
    :rtype: np.ndarray
    """
    if line_type == "gaussian":
        line_model = get_pixel_integrated_gaussian(
            x,
            mu,
            fwhm,
            gaussian_amp,
            continuum_amp,
            continuum_slope,
        )
    elif line_type == "voigt":
        line_model = get_pixel_integrated_voigt(
            x,
            mu,
            fwhm,
            voigt_gamma,
            gaussian_amp,
            continuum_amp,
            continuum_slope,
        )
    elif line_type == "lorentzian":
        line_model = get_pixel_integrated_lorentzian(
            x,
            mu,
            fwhm,
            gaussian_amp,
            continuum_amp,
            continuum_slope,
        )
    else:
        raise ValueError(f"Unknown line type: {line_type}")

    return line_model


def best_linear_fit_model(
    velocity,
    fwhms,
    wavelengths,
    spectra,
    noise,
    lines,
    line_type="gaussian",
    voigt_gamma=1.0,
    get_amp=False,
):
    """Best linear fit model for the spectra.

    :param params: parameters for the model
    :type params: list
    :param wavelengths: wavelengths
    :type wavelengths: np.ndarray
    :param spectra: spectra
    :type spectra: np.ndarray
    :param noise: noise
    :type noise: np.ndarray
    :param lines: lines
    :type lines: list
    :return: line model
    :rtype: np.ndarray
    """
    if isinstance(fwhms, float):
        fwhms = np.ones_like(lines, dtype=float) * fwhms

    line_models = []

    for fwhm, line in zip(fwhms, lines):
        mu = line * (1 + velocity / light_speed)
        line_model = get_spectra_model(
            wavelengths, mu, fwhm, line_type=line_type, voigt_gamma=voigt_gamma
        )
        line_models.append(line_model)

    line_models.append(np.ones_like(line_model))
    line_models.append(np.arange(len(line_model)))

    A = np.array(line_models).T

    b = spectra
    w = 1 / noise**2

    A_weighted = A * np.sqrt(w)[:, np.newaxis]
    b_weighted = b * np.sqrt(w)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
    except np.linalg.LinAlgError:
        # If the matrix is singular, use nnls
        coeffs, _ = nnls(A_weighted, b_weighted)

    line_model = A @ coeffs

    if get_amp:
        return line_model, coeffs
    else:
        return line_model


def best_linear_fit_model_simultaneous(
    velocity_1,
    fwhms_1,
    velocity_2,
    fwhms_2,
    wavelengths_1,
    spectra_1,
    noise_1,
    wavelengths_2,
    spectra_2,
    noise_2,
    lines,
):
    if isinstance(fwhms_1, float):
        fwhms_1 = np.ones_like(lines, dtype=float) * fwhms_1
    if isinstance(fwhms_2, float):
        fwhms_2 = np.ones_like(lines, dtype=float) * fwhms_2

    line_models_1 = []
    line_models_2 = []

    for fwhm_1, fwhm_2, line in zip(fwhms_1, fwhms_2, lines):
        mu_1 = line * (1 + velocity_1 / light_speed)
        # sigma_1 = fwhm_1 / 2.355
        line_model_1 = get_spectra_model(wavelengths_1, mu_1, fwhm_1)
        line_models_1.append(line_model_1)

        mu_2 = line * (1 + velocity_2 / light_speed)
        # sigma_2 = fwhm_2 / 2.355
        line_model_2 = get_spectra_model(wavelengths_2, mu_2, fwhm_2)
        line_models_2.append(line_model_2)

    line_models_1.append(np.ones_like(line_model_1))
    line_models_1.append(np.arange(len(line_model_1)))

    # convert to numpy array
    A = np.array(line_models_1).T

    b = spectra_1
    w = 1 / noise_1**2

    A_weighted = A * np.sqrt(w)[:, np.newaxis]
    b_weighted = b * np.sqrt(w)

    coeffs, _ = nnls(A_weighted, b_weighted)  # , rcond=None)

    # print(A.shape, coeffs.shape, b.shape)
    # line_model = A @ coeffs
    line_model_1 = A @ coeffs

    spec_model_2 = np.array(line_models_2).T @ coeffs[:-2]

    A_2 = np.vstack(
        (spec_model_2, np.ones_like(spec_model_2), np.arange(len(spec_model_2)))
    ).T

    b_2 = spectra_2
    w_2 = 1 / noise_2**2
    A_weighted_2 = A_2 * np.sqrt(w_2)[:, np.newaxis]
    b_weighted_2 = b_2 * np.sqrt(w_2)
    coeffs_2, _ = nnls(A_weighted_2, b_weighted_2)  # , rcond=None)

    line_model_2 = A_2 @ coeffs_2

    return line_model_1, line_model_2


def get_spectra_cuts(start, end, wavelengths, spectra_1d, noise_1d):
    """
    Get the spectra cut for the ith line.

    :param start: start wavelength
    :type start: float
    :param end: end wavelength
    :type end: float
    :return: wavelengths, spectra, noise
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    wavelengths_cut = wavelengths[(wavelengths > start) & (wavelengths < end)]
    spectra_cut = spectra_1d[(wavelengths > start) & (wavelengths < end)]
    noise_cut = noise_1d[(wavelengths > start) & (wavelengths < end)]

    return wavelengths_cut, spectra_cut, noise_cut
