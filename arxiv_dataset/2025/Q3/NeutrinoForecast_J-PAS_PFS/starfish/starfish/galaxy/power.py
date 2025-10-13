import camb
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from astropy.constants import c
import astropy.units as u

from ..set import *

__all__ = ["get_res_lcdm", "get_res_w0wacdm", "P_lin_lcdm", "P_lin_w0wacdm"]


def get_res_lcdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff):
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
    )
    pars.InitPower.set_params(ns=n_s)
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

    pars.set_matter_power(
        redshifts=settings.z,
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=settings.nonlinear,
        accurate_massive_neutrino_transfers=True,
        silent=True,
    )
    res = camb.get_results(pars)

    return pars, res


def get_res_w0wacdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, w0, wa):
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
    )
    pars.InitPower.set_params(ns=n_s)
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

    pars.set_matter_power(
        redshifts=settings.z,
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=settings.nonlinear,
        accurate_massive_neutrino_transfers=True,
        silent=True,
    )
    res = camb.get_results(pars)

    return pars, res


def P_lin_lcdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff):
    """
    linear power spectrum
    """

    pars, res = get_res_lcdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff)
    # interpolator in Mpc^3
    PK = res.get_matter_power_interpolator(
        nonlinear=settings.nonlinear,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )

    return pars, res, PK


def P_lin_w0wacdm(omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, w0, wa):
    """
    linear power spectrum
    """

    pars, res = get_res_w0wacdm(
        omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, w0, wa
    )
    # interpolator in Mpc^3
    PK = res.get_matter_power_interpolator(
        nonlinear=settings.nonlinear,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )

    return pars, res, PK


"""
# now all in format of 3-dimensional arrays
"""


def check_3dim(x, req):
    """
    shape
    - k and function of k: (1,-1,1)
    - z and function of z: (-1,1,1)
    - mu and function of mu: (1,1,-1)
    """

    x = np.array(x)
    check = dict(
        zip(("z", "k", "mu"), (len(settings.z), settings.k_points, settings.mu_points))
    )  # dictionary of lengths of axes

    if x.ndim <= 1:  # simple scalar or 1-d array, independent of other variables
        if req not in check.keys():
            raise ValueError(
                "requirements not satisfied! must be one of ('z','k','mu')"
            )
        else:
            index = [1, 1, 1]
            for i, key in enumerate(check.keys()):
                if req == key:
                    index[i] = -1
                    return x.reshape(
                        index
                    )  # reshape 1-d x into array of a desired shape
    elif x.ndim == 2:  # 2-d arrays are not expected as input
        raise ValueError("array k should not be 2-dimensional!")
    else:  # 3-d array, dependent of other variables
        if req not in check.keys():
            raise ValueError(
                "requirements not satisfied! must be one of ('z','k','mu')"
            )
        else:
            for i, key in enumerate(check.keys()):
                if req == key:
                    # check the shape of x along the desired axis
                    if x.shape[i] == check[key]:
                        return x
                    else:
                        raise ValueError(
                            f"shape of array x (axis {i}: {x.shape[i]}) not satisfied with requirements ({check[key]})!"
                        )


def P_smooth(k_eval, z_eval, PK, smooth=False, window_length=settings.window_len):
    """
    (not) smoothed power spectrum
    """

    if not np.all(np.in1d(z_eval, settings.z)):
        raise ValueError(
            f"z_eval ({z_eval}) contains elements that do not belong to z ({settings.z}). PK cannot return inaccurate extrapolated results!"
        )

    k_eval = check_3dim(k_eval, "k")
    z_eval = check_3dim(z_eval, "z")

    power_spec = PK.P(z_eval, k_eval, grid=False) * settings.h**3  # in (Mpc/h)^3
    if smooth:
        power_spec = savgol_filter(power_spec, window_length, polyorder=3, axis=1)

    return power_spec


"""
$$\sigma^2_\mathrm{p}(z)=\frac1{6\pi^2}\int\text{d}k P_{\text{cb,lin}}{(k,z)}$$
"""


def sigma2_p(PK):
    res = []
    abserr = []

    for i in range(len(settings.z)):
        r, e = quad(
            P_smooth,
            a=settings.k_min,
            b=np.max(settings.k_max),
            args=(settings.z[i], PK, False),
            limit=100,
        )
        res.append(r / (6 * np.pi**2))
        abserr.append(e)

    return check_3dim(res, "z"), check_3dim(abserr, "z")


"""
$$
f(z,k)=\frac12\frac{\mathrm{d}\ln P_\mathrm{cc}(z,k)}{\mathrm{d}\ln a}=-\frac{1+z}{2P_\mathrm{cc}}\frac{\mathrm{d}P_\mathrm{cc}(z,k)}{\mathrm{d}z}
$$
"""


def get_growth(pars):
    """
    growth rate
    """

    k = check_3dim(
        np.logspace(
            np.log10(settings.k_min) - 1.5,
            np.log10(np.max(settings.k_max)) + 1.5,
            num=2 * settings.k_points,
        ),
        "k",
    )

    pars.set_matter_power(
        redshifts=settings.z,
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=False,
        accurate_massive_neutrino_transfers=True,
        silent=True,
    )
    pars_minus = pars.copy()
    pars_minus.set_matter_power(
        redshifts=settings.z_min,
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=False,
        accurate_massive_neutrino_transfers=True,
        silent=True,
    )
    pars_plus = pars.copy()
    pars_plus.set_matter_power(
        redshifts=settings.z_max,
        kmax=settings.k_max_transfer,
        k_per_logint=settings.k_per_logint,
        nonlinear=False,
        accurate_massive_neutrino_transfers=True,
        silent=True,
    )

    res = camb.get_results(pars)
    res_minus = camb.get_results(pars_minus)
    res_plus = camb.get_results(pars_plus)

    PK_fid = res.get_matter_power_interpolator(
        nonlinear=False,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )
    PK_minus = res_minus.get_matter_power_interpolator(
        nonlinear=False,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )
    PK_plus = res_plus.get_matter_power_interpolator(
        nonlinear=False,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )

    pks_fid = (
        PK_fid.P(check_3dim(settings.z, "z"), k, grid=False) * settings.h**3
    )  # in (Mpc/h)^3
    pks_minus = (
        PK_minus.P(check_3dim(settings.z_min, "z"), k, grid=False) * settings.h**3
    )  # in (Mpc/h)^3
    pks_plus = (
        PK_plus.P(check_3dim(settings.z_max, "z"), k, grid=False) * settings.h**3
    )  # in (Mpc/h)^3

    partial_p_z = (pks_plus - pks_minus) / (
        check_3dim(settings.z_max, "z") - check_3dim(settings.z_min, "z")
    )
    f = -(1 + check_3dim(settings.z, "z")) / (2 * pks_fid) * partial_p_z
    interp = RectBivariateSpline(settings.z, np.log10(k.flatten()), f)

    return interp


def calc_growth(f_interp, k_eval, z_eval):
    if not np.all(np.in1d(z_eval, settings.z)):
        raise ValueError(
            f"z_eval ({z_eval}) contains elements that do not belong to z ({settings.z}). f cannot return inaccurate extrapolated results!"
        )

    k_eval = check_3dim(k_eval, "k")
    z_eval = check_3dim(z_eval, "z")

    return f_interp(z_eval, np.log10(k_eval), grid=False)


"""
$P_{\mathrm{obs}}(k_{\mathrm{fid}},\mu_{\mathrm{fid}};z)=\frac{1}{q_\perp^2(z)q_\parallel(z)}\left\{\frac{\left[b\sigma_8(z)+f(k,z)\sigma_8(z)\mu^2\right]^2}{1+f(k,z)^2k^2\mu^2\sigma_\mathrm{p}^2(z)}\right\}\frac{P_{\mathrm{dw}}(k,\mu;z)}{\sigma_8^2(z)}F_z(k,\mu;z)+P_\mathrm{s}(z)$
"""


def P_nonlin(k, mu, res, f_interp, PK, s2_p_fid, f_fid_interp):
    """
    observed power spectrum
    """

    k = check_3dim(k, req="k")
    mu = check_3dim(mu, req="mu")
    lnbs_value = np.loadtxt("lnbs.csv")  # in single-tracer case should be 1-dimensional
    if lnbs_value.ndim == 1:
        bs_8 = check_3dim(np.exp(lnbs_value), "z")
    elif lnbs_value.ndim == 2:
        bs_8 = np.exp(lnbs_value).reshape(
            (len(settings.z), 1, 1, len(settings.tracers))
        )
    else:
        raise ValueError(
            f"lnbs.csv should be 1- or 2-dimensional, not {lnbs_value.ndim}-dimensional!"
        )
    s_8 = check_3dim(np.flip(res.get_sigmaR(8, None, "delta_nonu", "delta_nonu")), "z")
    s2_v_fid = s2_p_fid

    f = calc_growth(f_interp, k, settings.z)
    f_fid = calc_growth(f_fid_interp, k, settings.z)

    if lnbs_value.ndim == 1:
        rsd = (bs_8 + f * s_8 * mu**2) ** 2
    else:
        temp = f * s_8 * mu**2
        rsd = (bs_8 + temp.reshape(temp.shape + (1,))) ** 2
    fog = 1 + (f_fid**2) * (k**2) * (mu**2) * s2_p_fid
    g_mu = s2_v_fid * (1 - mu**2 + mu**2 * (1 + f_fid) ** 2)

    pks_lin = P_smooth(k, settings.z, PK, smooth=False)
    pks_nw = P_smooth(k, settings.z, PK, smooth=True, window_length=settings.window_len)

    e_factor = np.exp(-g_mu * k**2)
    pks_dw = pks_lin * e_factor + pks_nw * (1 - e_factor)
    if np.isscalar(settings.sigma_0z):
        s_0z = settings.sigma_0z
        sigma_r = (
            (c.to(u.km / u.s)).value
            * (1 + check_3dim(settings.z, "z"))
            * s_0z
            / check_3dim(res.hubble_parameter(settings.z), "z")
        )
        F_z = np.exp(-((k * mu * sigma_r) ** 2))
        if lnbs_value.ndim == 1:
            pass  # 3-dimensional
        else:
            F_z = F_z.reshape(F_z.shape + (1,))  # 4-dimensional
    else:
        s_0z = np.array(settings.sigma_0z).reshape(1, 1, 1, -1)
        sigma_r = (
            (c.to(u.km / u.s)).value
            * (1 + np.array(settings.z).reshape(-1, 1, 1, 1))
            * s_0z
            / np.reshape(res.hubble_parameter(settings.z), (-1, 1, 1, 1))
        )
        F_z = np.exp(
            -((k.reshape(k.shape + (1,)) * mu.reshape(mu.shape + (1,)) * sigma_r) ** 2)
        )  # 4-dimensional

    if lnbs_value.ndim == 1:
        return rsd / fog * pks_dw / s_8**2 * F_z
    else:
        temp = 1 / fog * pks_dw / s_8**2
        return rsd * temp.reshape(temp.shape + (1,)) * F_z


def P_AP(res, f_interp, PK, H_fid, DA_fid, s2_p_fid, f_fid_interp):
    """
    observed power spectrum with AP
    """

    k = check_3dim(settings.k_fid, "k")
    mu = check_3dim(settings.mu_fid, "mu")

    H = check_3dim(res.hubble_parameter(settings.z), "z")
    DA = check_3dim(res.angular_diameter_distance(settings.z), "z")

    q_par = H_fid / H
    q_per = DA / DA_fid
    G = np.sqrt(1 + mu**2 * ((q_per / q_par) ** 2 - 1))

    if (
        not hasattr(settings, "tracers")
        or np.isscalar(settings.tracers)
        or len(settings.tracers) == 1
    ):
        out = (
            1
            / (q_par * q_per**2)
            * P_nonlin(
                k * G / q_per,
                mu * q_per / q_par / G,
                res,
                f_interp,
                PK,
                s2_p_fid,
                f_fid_interp,
            )
            + 0
        )

        if np.isscalar(settings.k_max):
            pass
        else:
            for i in range(len(settings.z)):
                pivot = int(
                    (
                        (np.log10(settings.k_max[i]) - np.log10(settings.k_min))
                        / (np.log10(np.max(settings.k_max)) - np.log10(settings.k_min))
                    )
                    * (settings.k_points - 1)
                )
                out[i, pivot + 1 :, :] = 0
    else:
        out = (
            1
            / (q_par * q_per**2).reshape(-1, 1, 1, 1)
            * P_nonlin(
                k * G / q_per,
                mu * q_per / q_par / G,
                res,
                f_interp,
                PK,
                s2_p_fid,
                f_fid_interp,
            )
            + 0
        )

        if np.isscalar(settings.k_max):
            pass
        else:
            for i in range(len(settings.z)):
                pivot = int(
                    (
                        (np.log10(settings.k_max[i]) - np.log10(settings.k_min))
                        / (np.log10(np.max(settings.k_max)) - np.log10(settings.k_min))
                    )
                    * (settings.k_points - 1)
                )
                out[i, pivot + 1 :, :, :] = 0

    return out
