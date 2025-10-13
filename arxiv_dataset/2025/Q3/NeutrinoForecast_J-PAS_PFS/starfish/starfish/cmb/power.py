import camb
import numpy as np

from ..set import *

__all__ = [
    "get_res_lcdm",
    "get_res_w0wacdm",
    "C_ell_lcdm",
    "C_ell_w0wacdm",
    "get_noise",
]


def get_res_lcdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau):
    """
    cosmological parameters initialization
    """

    omnuh2 = m_nu / 94.07 * (N_eff / 3) ** (3 / 4)

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * h,
        ombh2=omega_b0 * h**2,
        omch2=(omega_m0 - omega_b0) * h**2 - omnuh2,
        omk=0,
        num_massive_neutrinos=settings.nu_mass_num,
        mnu=m_nu,
        nnu=N_eff,
        tau=tau,
    )
    pars.InitPower.set_params(ns=n_s)
    pars.set_for_lmax(lmax=np.max((settings.ell_max,3000)), lens_potential_accuracy=settings.lens_accuracy)
    pars.set_matter_power(
        redshifts=[0],
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=settings.nonlinear,
        accurate_massive_neutrino_transfers=True,
    )
    pars.NonLinearModel.halofit_version = settings.halofit_ver
    pars.set_accuracy(
        AccuracyBoost=settings.AccuracyBoost, lAccuracyBoost=settings.lAccuracyBoost
    )

    # scale sigma_8 of *matter*
    A_s = pars.InitPower.As * (sigma_8 / camb.get_results(pars).get_sigma8_0()) ** 2
    pars.InitPower.set_params(As=A_s, ns=n_s)

    res = camb.get_results(pars)
    return pars, res


def get_res_w0wacdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau, w0, wa):
    """
    cosmological parameters initialization
    """

    omnuh2 = m_nu / 94.07 * (N_eff / 3) ** (3 / 4)

    pars = camb.CAMBparams()
    pars.DarkEnergy = camb.DarkEnergyPPF(w=w0, wa=wa)
    pars.set_cosmology(
        H0=100 * h,
        ombh2=omega_b0 * h**2,
        omch2=(omega_m0 - omega_b0) * h**2 - omnuh2,
        omk=0,
        num_massive_neutrinos=settings.nu_mass_num,
        mnu=m_nu,
        nnu=N_eff,
        tau=tau,
    )
    pars.InitPower.set_params(ns=n_s)
    pars.set_for_lmax(lmax=np.max((settings.ell_max,3000)), lens_potential_accuracy=settings.lens_accuracy)
    pars.set_matter_power(
        redshifts=[0],
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=settings.nonlinear,
        accurate_massive_neutrino_transfers=True,
    )
    pars.NonLinearModel.halofit_version = settings.halofit_ver
    pars.set_accuracy(
        AccuracyBoost=settings.AccuracyBoost, lAccuracyBoost=settings.lAccuracyBoost
    )

    # scale sigma_8 of *matter*
    A_s = pars.InitPower.As * (sigma_8 / camb.get_results(pars).get_sigma8_0()) ** 2
    pars.InitPower.set_params(As=A_s, ns=n_s)

    res = camb.get_results(pars)
    return pars, res


def C_ell_lcdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau):
    """
    $C_\ell$ of CMB
    """

    pars, res = get_res_lcdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau)
    # unit in muK, raw C_\ell
    cls = res.get_total_cls(lmax=np.max((settings.ell_max,3000)), CMB_unit="muK", raw_cl=True)

    return pars, res, cls[:, [0, 1, 3]]


def C_ell_w0wacdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau, w0, wa):
    """
    $C_\ell$ of CMB
    """

    pars, res = get_res_w0wacdm(
        omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau, w0, wa
    )
    # unit in muK, raw C_\ell
    cls = res.get_total_cls(lmax=np.max((settings.ell_max,3000)), CMB_unit="muK", raw_cl=True)

    return pars, res, cls[:, [0, 1, 3]]


"""
$$
N^\alpha_\ell=\sigma_\alpha^2\exp\left[\frac{\ell(\ell+1)\theta^2_\mathrm{FWHM}}{8\ln2}\right]
$$
"""


def get_noise():
    """
    noise $N_\ell$
    """

    ells = np.arange(np.max((settings.ell_max,3000)) + 1)

    s = np.zeros_like(ells, dtype=float)
    for i in range(len(settings.FWHM)):
        s += 1 / (
            settings.sigma2_T[i]
            * np.exp(ells * (ells + 1) * settings.FWHM[i] ** 2 / (8 * np.log(2)))
        )
    N_T = 1 / s

    s = np.zeros_like(ells, dtype=float)
    for i in range(len(settings.FWHM)):
        s += 1 / (
            settings.sigma2_P[i]
            * np.exp(ells * (ells + 1) * settings.FWHM[i] ** 2 / (8 * np.log(2)))
        )
    N_P = 1 / s

    return N_T, N_P
