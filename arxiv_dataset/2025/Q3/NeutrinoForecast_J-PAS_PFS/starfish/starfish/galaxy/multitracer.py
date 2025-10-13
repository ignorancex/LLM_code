from .power import (
    P_lin_lcdm,
    P_lin_w0wacdm,
    P_nonlin,
    P_AP,
    get_growth,
    calc_growth,
    check_3dim,
    sigma2_p,
)
import os
import sys
import shutil
import camb
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import gc
from ..set import *

powers = 0
V_fid = []

__all__ = ["powers", "V_fid", "get_volume", "get_fid_values", "get_Fisher"]


def get_volume(res):
    """
    volume $V(z)$
    """

    chi_min = res.angular_diameter_distance(settings.z_min) * (
        1 + settings.z_min
    )  # in Mpc
    chi_max = res.angular_diameter_distance(settings.z_max) * (
        1 + settings.z_max
    )  # same as $\int c dz/H(z)$
    A_sphere = 4 * np.pi * (180 / np.pi) ** 2
    f_sky = settings.A_sky / A_sphere
    return (
        f_sky * 4 * np.pi / 3 * (chi_max**3 - chi_min**3) * settings.h**3
    )  # in (Mpc/h)^3


def get_fid_values(lcdm=True):
    """
    fiducial nuisance parameters and $H(z)$, $D_\mathrm{A}(z)$, $f(z,k)$
    """

    if lcdm:
        pars, res, PK_fid = P_lin_lcdm(*settings.cosmo_value)
        pars0 = pars.copy()
        # set redshift of pars0 from default z to [0], with that of pars and res unchanged
        pars0.set_matter_power(
            redshifts=[0],
            kmax=settings.k_max_transfer,
            k_per_logint=settings.k_per_logint,
            nonlinear=settings.nonlinear,
            accurate_massive_neutrino_transfers=True,
        )
        fid_compute = [
            pars0.omegam,
            pars0.omegab,
            pars0.h,
            pars0.InitPower.ns,
            camb.get_results(pars0).get_sigma8_0(),
            pars0.omnuh2 / (pars0.N_eff / 3) ** (3 / 4) * 94.07,
            pars0.N_eff,
        ]
    else:
        pars, res, PK_fid = P_lin_w0wacdm(*settings.cosmo_value)
        pars0 = pars.copy()
        pars0.set_matter_power(
            redshifts=[0],
            kmax=settings.k_max_transfer,
            k_per_logint=settings.k_per_logint,
            nonlinear=settings.nonlinear,
            accurate_massive_neutrino_transfers=True,
        )
        fid_compute = [
            pars0.omegam,
            pars0.omegab,
            pars0.h,
            pars0.InitPower.ns,
            camb.get_results(pars0).get_sigma8_0(),
            pars0.omnuh2 / (pars0.N_eff / 3) ** (3 / 4) * 94.07,
            pars0.N_eff,
            pars0.DarkEnergy.w,
            pars0.DarkEnergy.wa,
        ]

    global V_fid
    V_fid = get_volume(res)
    print("fiducial volumes:")
    for i in range(len(settings.z)):
        print(f"V_fid ({settings.z[i]:.1f}) = {V_fid[i]:.2e}")
    print("fiducial biases:")
    for t in range(len(settings.tracers)):
        print("#" * 20 + f" tracer: {settings.tracers[t]} " + "#" * 20)
        for i in range(len(settings.z)):
            print(f"b_fid ({settings.z[i]:.1f}) = {settings.b_fid[i,t]:.2f}")

    name_width = max([len(v) for v in settings.var_name])
    nuisance_compute = np.log(
        np.array(settings.b_fid)
        * np.flip(res.get_sigmaR(8, None, "delta_nonu", "delta_nonu")).reshape((-1, 1))
    )
    np.savetxt("lnbs.csv", nuisance_compute)
    fid_dict = dict(
        zip(
            settings.var_name,
            np.concatenate(
                (
                    fid_compute,
                    np.zeros(len(settings.tracers) * len(settings.z)),
                    nuisance_compute.flatten(),
                )
            ),
        )
    )
    print("fiducial values:")
    for key in fid_dict.keys():
        print(f"-- {key.rjust(name_width)}: {fid_dict[key]}")

    print("calculating fiducial H(z), DA(z) and f(z,k)...  ", end="")
    H_fid = res.hubble_parameter(settings.z)
    DA_fid = res.angular_diameter_distance(settings.z)
    s2_p, abserr = sigma2_p(PK_fid)
    thres = 1e-6
    if np.any(abserr / s2_p > thres):
        raise ValueError(f"relative errors of sigma2_p are above {thres}!")
    print("done!")

    return res, check_3dim(H_fid, "z"), check_3dim(DA_fid, "z"), s2_p, get_growth(pars)


def get_power_obs(cosmo_params, H_fid, DA_fid, s2_p_fid, f_fid_interp, lcdm=True):
    """
    overall function returning $P_\mathrm{obs}(z_i,k,\mu;\boldsymbol{\theta})$
    """

    if lcdm:
        pars, res, PK = P_lin_lcdm(*cosmo_params)
    else:
        pars, res, PK = P_lin_w0wacdm(*cosmo_params)
    f_interp = get_growth(pars)
    pks = P_AP(
        res, f_interp, PK, H_fid, DA_fid, s2_p_fid, f_fid_interp
    )  # 4-dimensional
    global powers
    powers += 1
    print(
        f"({powers:02d}/{2*settings.cosmo_num}) a power spectrum of one set of parameters done!"
    )

    return pks


"""
# 2 types of partial derivative algorithm
"""


def calc_partial_simple(func, x0, step, args=None):
    x0 = np.array([x0], dtype=float) if np.isscalar(x0) else np.array(x0, dtype=float)
    step = (
        np.array([step], dtype=float)
        if np.isscalar(step)
        else np.array(step, dtype=float)
    )

    if x0.ndim != 1 or step.ndim != 1:
        raise ValueError(
            f"only 1-d x0 (ndim: {x0.ndim}) and step (ndim: {step.ndim}) are accepted by `calc_partial_nd`!"
        )
    if x0.shape != step.shape:
        raise ValueError(
            f"shape of x0 {x0.shape} dose not match that of step {step.shape}!"
        )
    size = x0.size

    def diff(func, x_plus, x_minus, args):
        if args == None:
            f_plus = func(x_plus)
            f_minus = func(x_minus)
        elif np.isscalar(args):
            f_plus = func(x_plus, args)
            f_minus = func(x_minus, args)
        else:
            f_plus = func(x_plus, *args)
            f_minus = func(x_minus, *args)
        return f_plus - f_minus

    df_dx = []
    if size == 1:
        x_plus = (x0 + step)[0]
        x_minus = (x0 - step)[0]
        df_dx.append(diff(func, x_plus, x_minus, args) / (2 * step[0]))
    else:
        for i in range(size):
            x_plus = x0.copy()
            x_minus = x0.copy()

            x_plus[i] += step[i]
            x_minus[i] -= step[i]

            df_dx.append(diff(func, x_plus, x_minus, args) / (2 * step[i]))

    return np.stack(df_dx, axis=-1)


"""
$$
\left.\frac{\partial\ln P_\text{obs}}{\partial p_\text{s}}\right|_{\text{fid},z=z_i}=\left.\frac1{P_\text{obs}}\right|_{\text{fid},z=z_i}
$$

$$
\left.\frac{\partial\ln P_\text{obs}}{\partial\ln(b\sigma_8)}\right|_{\text{fid},z=z_i}=\left.\frac{2b}{b+f(z,k)\mu_\text{obs}^2}\right|_{\text{fid},z=z_i}
$$

$$
\left.\frac{\partial\ln P_\text{obs}}{\partial\sigma^2_\mathrm{p}}\right|_{\text{fid},z=z_i}=\left.-\frac{f(z,k)^2k^2\mu^2}{1+f(z,k)^2k^2\mu^2\sigma^2_\mathrm{p}}\right|_{\text{fid},z=z_i}
$$

$$
\left.\frac{\partial\ln P_\text{obs}}{\partial\sigma^2_\mathrm{v}}\right|_{\text{fid},z=z_i}=\left.\frac{P_\mathrm{nw}-P_\mathrm{cc}}{P_\mathrm{dw}}\exp(-g_\mu k^2)k^2\left[1-\mu^2+\mu^2(1+f^\mathrm{fid}(z,k))^2\right]\right|_{\text{fid},z=z_i}
$$

"""


def get_partial(lcdm=True):
    """
    integrand of Fisher matrix
    """

    res, *fid = get_fid_values(lcdm)
    [H_fid, DA_fid, s2_p_fid, f_fid_interp] = fid
    f_fid = calc_growth(f_fid_interp, settings.k_fid, settings.z)
    # interpolator in Mpc^3
    PK_fid = res.get_matter_power_interpolator(
        nonlinear=settings.nonlinear,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )
    pks_fid = P_nonlin(
        settings.k_fid,
        settings.mu_fid,
        res,
        f_fid_interp,
        PK_fid,
        s2_p_fid,
        f_fid_interp,
    )
    print("all necessary fiducial values done!")

    print("calculating partial derivatives w.r.t cosmological parameters...")
    if lcdm:
        step = 1e-2 * np.array(settings.cosmo_value)
    else:
        step = np.concatenate(
            (1e-2 * np.array(settings.cosmo_value[:-2]), [0.01, 0.01])
        )
    (index_nu,) = np.where(np.array(settings.var_name) == "m_nu")
    step[index_nu[0]] = 0.1 * settings.m_nu
    global powers
    powers = 0
    cosmo_grad = calc_partial_simple(
        get_power_obs,
        settings.cosmo_value,
        step,
        fid
        + [
            lcdm,
        ],
    ) / pks_fid.reshape(
        pks_fid.shape + (1,)
    )  # 5-dimensional here
    print("partial derivatives w.r.t cosmological parameters done!")

    print(
        "calculating and saving partial derivatives into temporary path: "
        + settings.temp_path,
    )
    os.makedirs(settings.temp_path, exist_ok=True)
    n = 0
    for i in range(settings.cosmo_num):
        np.save(
            settings.temp_path + f"partial_{i:03d}.npy",
            cosmo_grad[:, :, :, :, i],
        )
        n += 1
        sys.stdout.write(f"\rprogress: {n}/{settings.var_num}")
        sys.stdout.flush()

    del cosmo_grad
    gc.collect()
    ps_grad = np.zeros(
        (
            len(settings.z),
            settings.k_points,
            settings.mu_points,
            len(settings.tracers),
            len(settings.tracers) * len(settings.z),
        )
    )
    for i in range(len(settings.z)):
        for t in range(len(settings.tracers)):
            ps_grad[i, :, :, t, len(settings.tracers) * i + t] = 1 / pks_fid[i, :, :, t]
            if np.isscalar(settings.k_max):
                pass
            else:
                pivot = int(
                    (
                        (np.log10(settings.k_max[i]) - np.log10(settings.k_min))
                        / (np.log10(np.max(settings.k_max)) - np.log10(settings.k_min))
                    )
                    * (settings.k_points - 1)
                )
                ps_grad[i, pivot + 1 :, :, :, :] = 0
    for i in range(len(settings.z)):
        for t in range(len(settings.tracers)):
            np.save(
                settings.temp_path
                + f"partial_{settings.cosmo_num+len(settings.tracers)*i+t:03d}.npy",
                ps_grad[:, :, :, :, len(settings.tracers) * i + t],
            )
            n += 1
            sys.stdout.write(f"\rprogress: {n}/{settings.var_num}")
            sys.stdout.flush()

    del ps_grad
    gc.collect()
    lnbs_grad = np.zeros(
        (
            len(settings.z),
            settings.k_points,
            settings.mu_points,
            len(settings.tracers),
            len(settings.tracers) * len(settings.z),
        )
    )
    for i in range(len(settings.z)):
        if np.isscalar(settings.k_max):
            pivot = settings.k_points
        else:
            pivot = int(
                (
                    (np.log10(settings.k_max[i]) - np.log10(settings.k_min))
                    / (np.log10(np.max(settings.k_max)) - np.log10(settings.k_min))
                )
                * (settings.k_points - 1)
            )
        for j in range(pivot):
            for t in range(len(settings.tracers)):
                lnbs_grad[i, j, :, t, len(settings.tracers) * i + t] = (
                    2
                    * settings.b_fid[i, t]
                    / (settings.b_fid[i, t] + f_fid[i, j, 0] * settings.mu_fid**2)
                )
    for i in range(len(settings.z)):
        for t in range(len(settings.tracers)):
            np.save(
                settings.temp_path
                + f"partial_{settings.cosmo_num+len(settings.tracers)*len(settings.z)+len(settings.tracers)*i+t:03d}.npy",
                lnbs_grad[:, :, :, :, len(settings.tracers) * i + t],
            )
            n += 1
            sys.stdout.write(f"\rprogress: {n}/{settings.var_num}")
            sys.stdout.flush()
    # g = np.concatenate((cosmo_grad, ps_grad, lnbs_grad), axis=-1)

    print("\npartial derivatives done!")
    return pks_fid  # shape of (z_num, k_num, mu_num, tracers_num, var_num)


"""
$$F_{\alpha \beta}=\left.\left.\sum_i \frac{1}{8 \pi^2} \int_{-1}^1 \mathrm{~d} \mu \int_{10^{-4}}^{0.5} \mathrm{~d} k k^2 \partial_\alpha \ln \left(P_{\text {obs }}\right)\right|_{\mathrm{fid}} \partial_\beta \ln \left(P_{\text {obs }}\right)\right|_{\mathrm{fid}} V_i^{\mathrm{eff}}$$
"""


def get_Fisher(lcdm=True):
    print("process started!")
    pks_fid = get_partial(lcdm)
    # pks_fid = np.ones(
    #     (len(settings.z), settings.k_points, settings.mu_points, len(settings.tracers))
    # )

    # Fisher information density
    P_tracer = pks_fid * settings.n_fid.reshape(
        len(settings.z), 1, 1, len(settings.tracers)
    )
    P = np.sum(P_tracer, axis=-1)  # 3-dimentional

    density = np.zeros(
        (
            len(settings.z),
            settings.k_points,
            settings.mu_points,
            len(settings.tracers),
            len(settings.tracers),
        )
    )
    others = (1 - P) / (1 + P) ** 2
    for a in range(len(settings.tracers)):
        for b in range(len(settings.tracers)):
            if a != b:
                density[:, :, :, a, b] = (
                    P_tracer[:, :, :, a] * P_tracer[:, :, :, b] * others
                )
            else:
                density[:, :, :, a, b] = P_tracer[:, :, :, a] * P_tracer[
                    :, :, :, b
                ] * others + P_tracer[:, :, :, a] * P / (1 + P)
    density /= 2

    # cross product and integrate
    Fisher_ij = np.zeros((len(settings.z), settings.var_num, settings.var_num))
    print("loading partial derivatives and finishing simpson integration...")

    n = 0
    iterations = int((1 + settings.var_num) * settings.var_num / 2)
    for i in range(settings.var_num):
        partial_i = np.load(settings.temp_path + f"partial_{i:03d}.npy")
        for j in range(i + 1):
            partial_j = np.load(settings.temp_path + f"partial_{j:03d}.npy")
            inside = 0
            for a in range(len(settings.tracers)):
                for b in range(len(settings.tracers)):
                    inside += (
                        partial_i[:, :, :, a]
                        * partial_j[:, :, :, b]
                        * density[:, :, :, a, b]
                    )
            # then in 3-dimensional
            inside *= check_3dim(V_fid, "z") * check_3dim(settings.k_fid**2, "k")
            int_k = simpson(inside, x=settings.k_fid, axis=1)
            int_k_mu = simpson(int_k, x=settings.mu_fid, axis=1) / (8 * np.pi**2)
            Fisher_ij[:, i, j] = int_k_mu
            Fisher_ij[:, j, i] = int_k_mu

            n += 1
            sys.stdout.write(f"\rprogress: {n}/{iterations:d}")
            sys.stdout.flush()

    print("\nintegration done!")

    print("deleting temporary files...")
    shutil.rmtree(settings.temp_path)
    print("deleting done!")

    os.makedirs(settings.save_path, exist_ok=True)
    if settings.save_path[-1] == "/":
        path = settings.save_path
    else:
        path = settings.save_path + "/"
    for i in range(len(settings.z)):
        np.savetxt(
            path + f"multitracer_{settings.z[i]:.1f}.csv",
            Fisher_ij[i, :, :],
            delimiter=",",
        )
        print(f"-- Fisher matrix of redshift {settings.z[i]:.1f} saved into " + path)

    df = pd.DataFrame(
        np.sum(Fisher_ij, axis=0), index=settings.var_name, columns=settings.var_name
    )
    df.to_excel(path + "fisher.xlsx")
    print(f"Fisher matrix saved into {path}fisher.xlsx!")

    print("all done!!!")
    return df
