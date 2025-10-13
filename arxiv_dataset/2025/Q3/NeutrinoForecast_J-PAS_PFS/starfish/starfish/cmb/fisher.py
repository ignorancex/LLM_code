from .power import *
from ..set import *
import numpy as np
import pandas as pd

powers = 0

__all__ = ["powers", "get_power", "get_cov", "get_fisher"]


def get_power(cosmo_params, lcdm=True):
    if lcdm:
        pars, res, cls = C_ell_lcdm(*cosmo_params)
    else:
        pars, res, cls = C_ell_w0wacdm(*cosmo_params)
    global powers
    powers += 1
    print(
        f"({powers:02d}/{2*settings.cosmo_num}) a power spectrum of one set of parameters done!"
    )

    return cls


"""
# covariance of $C_\ell$
"""


def get_cov(ell, cl_scalar_TEC, noise_T, noise_P):
    """
    in an order of (TT, EE, TE)
    """

    factor = (2 * ell + 1) * settings.f_sky

    cov = np.zeros((3, 3))
    cov[0, 0] = 2 / factor * (cl_scalar_TEC[0] + noise_T) ** 2
    cov[1, 1] = 2 / factor * (cl_scalar_TEC[1] + noise_P) ** 2
    cov[2, 2] = (
        1
        / factor
        * (
            cl_scalar_TEC[2] ** 2
            + (cl_scalar_TEC[0] + noise_T) * (cl_scalar_TEC[1] + noise_P)
        )
    )

    cov[0, 1] = cov[1, 0] = 2 / factor * cl_scalar_TEC[2] ** 2
    cov[0, 2] = cov[2, 0] = 2 / factor * cl_scalar_TEC[2] * (cl_scalar_TEC[0] + noise_T)
    cov[1, 2] = cov[2, 1] = 2 / factor * cl_scalar_TEC[2] * (cl_scalar_TEC[1] + noise_P)

    return cov


def calc_partial_simple(func, x0, step, args=None):
    """
    partial derivative algorithm
    """

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


def get_partial(lcdm=True):
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
    g = calc_partial_simple(get_power, settings.cosmo_value, step, args=lcdm)
    print("partial derivatives w.r.t cosmological parameters done!")

    return g


def get_fisher(lcdm=True):
    print("process started!")
    if lcdm:
        pars, res, cls = C_ell_lcdm(*settings.cosmo_value)
        fid_compute = [
            pars.omegam,
            pars.omegab,
            pars.h,
            pars.InitPower.ns,
            res.get_sigma8_0(),
            pars.omnuh2 / (pars.N_eff / 3) ** (3 / 4) * 94.07,
            pars.N_eff,
            pars.Reion.optical_depth,
        ]
    else:
        pars, res, cls = C_ell_w0wacdm(*settings.cosmo_value)
        fid_compute = [
            pars.omegam,
            pars.omegab,
            pars.h,
            pars.InitPower.ns,
            res.get_sigma8_0(),
            pars.omnuh2 / (pars.N_eff / 3) ** (3 / 4) * 94.07,
            pars.N_eff,
            pars.Reion.optical_depth,
            pars.DarkEnergy.w,
            pars.DarkEnergy.wa,
        ]
    name_width = max([len(v) for v in settings.var_name])
    fid_dict = dict(zip(settings.var_name, fid_compute))
    print("fiducial values:")
    for key in fid_dict.keys():
        print(f"-- {key.rjust(name_width)}: {fid_dict[key]}")

    p = get_partial(lcdm)
    N_T, N_P = get_noise()

    fisher = np.zeros((settings.cosmo_num, settings.cosmo_num))
    for i in range(settings.cosmo_num):
        for j in range(settings.cosmo_num):
            Fij_ell = []
            for ell in range(settings.ell_min, settings.ell_max + 1):
                cov = get_cov(ell, cls[ell, :], N_T[ell], N_P[ell])
                Fij_ell.append(
                    p[ell, :, i] @ np.linalg.inv(cov) @ p[ell, :, j].reshape((-1, 1))
                )
            fisher[i, j] = np.sum(Fij_ell)

    df = pd.DataFrame(fisher, index=settings.var_name, columns=settings.var_name)
    df.to_excel("fisher.xlsx")
    print("Fisher matrix saved into fisher.xlsx!")

    print("all done!!!")
    return df
