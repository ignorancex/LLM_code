# Author:
#          Mathieu Renzo <mrenzo@flatironinstitute.org>
#
# Keywords: files

# Copyright (C) 2020-2022 Mathieu Renzo

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.

import sys
import socket
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re
import os
import warnings
import glob
from math import pi

try:
    from termcolor import colored
except ImportError:

    def colored(a, color):
        return a


from lib_engineered import get_M_boundary

try:
    from MESAreader import getSrcCol, secyer, G_cgs, Lsun, Msun, Rsun_cm, clight
except:
    raise ModuleNotFoundError("MESAreader")

from scipy.interpolate import interp1d

def get_masses(f):
    """returns the initial masses of a MESA run reading them from the folder name
    It assumes the folder names follow the POSYDON convention

    f: string with the absolute path of the working directory
    output: m1, m2 (or m1, nan for a single star) as floats
    """
    if f[-1] == "/":
        folder = f.split("/")[-2]
    elif f[-1] != "/":
        folder = f.split("/")[-1]
    # print(folder)
    try:
        m = re.findall("[+-]?\d+\.\d+", folder)
        # print(m)
        if len(m) == 1:
            # print("single star!")
            m1 = float(m[0])
            m2 = np.nan
        else:
            # print("binary!")
            m1 = float(m[0])
            m2 = float(m[1])
    except:
        # read from src assuming single star
        src, col = getSrcCol(f + 'LOGS/history.data', False, False)
        m1 = src[0, col.index("star_mass")]
        ms = np.nan
    return m1, m2

# -------------------------------------------------------------------
# HRD
def plot_HRD(
    input_file,
    ax,
    convert=True,
    annotate_radii=None,
    annotate_TAMS=False,
    annotate_RLOF=False,
    **plot_kwargs,
):
    """Make HRD

    Parameters:
    ----------
    input_file: `string` or `os.path` of the history.data file
    convert: `bool` optional, create *.npy binary of the data file
    ax: `mpl.axes`
    annotate_radii: `None` or `np.array(dtype=float)`  optional, annotate loci of constant R provided
    annotate_TAMS:  `bool`, optional, mark TAMS
    annotate_RLOF:  `bool`, optional, try to highlight RLOF phase
    """
    # print(colored(input_file, "blue"))
    if os.path.isdir(input_file):
        src, col = getSrcCol(input_file + "/history.data", convert, convert)
    elif os.path.isfile(input_file):
        src, col = getSrcCol(input_file, convert, convert)
    else:
        raise FileNotFoundError(input_file)
    logT = src[:, col.index("log_Teff")]
    logL = src[:, col.index("log_L")]
    try:
        # annotate masses
        fff = input_file[:-7]
        m1, m2 = get_masses(fff)
        if "LOGS1" in input_file:
            ax.text(
                logT[0] - 0.01,
                logL[0] - 0.01,
                f"{m1:.1f}",
                fontsize=20,
                va="center",
                ha="center",
                transform=ax.transData,
            )
        elif "LOGS2" in folder:
            ax.text(
                logT[0] - 0.01,
                logL[0] - 0.01,
                f"{m2:.1f}",
                fontsize=20,
                va="center",
                ha="center",
                transform=ax.transData,
            )
    except:
        pass
    ax.plot(logT, logL, **plot_kwargs)
    if annotate_radii:
        annotate_radii_hrd(ax, radii=annotate_radii)
    if annotate_TAMS:
        X = src[:, col.index("center_h1")]
        iTAMS = np.argmin(np.absolute(X - 1e-4))
        try:
            color = plot_kwargs["c"]
        except:
            color = plot_kwargs["color"]
        ax.plot(
            logT[iTAMS],
            logL[iTAMS],
            lw=0,
            marker="o",
            mew=2,
            markeredgecolor="k",
            ms=10,
            color=color,
            zorder=10,
        )
    if annotate_RLOF:
        if os.path.isdir(input_file):
            bin_hfile = input_file + "../binary_history.data"
        elif os.path.isfile(input_file):
            if "LOGS1" in input_file:
                bin_hfile = input_file.replace(
                    "LOGS1/history.data", "binary_history.data"
                )
            elif "LOGS2" in input_file:
                bin_hfile = input_file.replace(
                    "LOGS2/history.data", "binary_history.data"
                )
        if os.path.isfile(bin_hfile):
            src, col = getSrcCol(bin_hfile, convert, convert)
            rl_relative_overflow_1 = src[:, col.index("rl_relative_overflow_1")]
            iRLOF = rl_relative_overflow_1 > 0
            # complete this array: no binary_history.data after detachment
            if len(logT) < len(iRLOF):
                # working on star 1
                iRLOF = iRLOF[
                    0 : len(logT)
                ]  # np.concatenate((iRLOF, np.full((len(iRLOF)-len(logT)), False)))
            ax.plot(
                logT[iRLOF],
                logL[iRLOF],
                color="y",
                lw=6,
                alpha=0.5,
                zorder=min(int(plot_kwargs["zorder"]) - 1, 0),
            )
        else:
            raise FileNotFoundError(bin_hfile)


def get_L_from_r_teff(radius, teff):
    # to annotate radii on HRD
    # Stephan Boltzman constant
    boltzm = 1.380649e-16  # cgs
    hbar = 6.62607015e-27 / (2 * pi)
    clight = 2.99792458e10
    sigma = (pi * pi * boltzm * boltzm * boltzm * boltzm) / (
        60 * hbar * hbar * hbar * clight * clight
    )
    # convert r to cm
    radius *= Rsun_cm
    # assume teff is in K
    l = 4 * pi * radius * radius * sigma * teff ** 4.0
    # convert to Lsun
    l = l / Lsun
    return l


def annotate_radii_hrd(ax, radii=np.logspace(0, 3, base=10)):
    """
    give the axis object for an HRD plot (assumed to be in log10 Lsun and log10 Teff),
    and a list of radii in Rsun units, plots radii.
    Parameters:
    ----------
    ax: `mpl.ax` matplotlib axis object
    radii: `np.array`, optional, radii to mark
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    Tmax = 10.0 ** xmax
    Tmin = 10.0 ** xmin
    teff = np.linspace(Tmin, Tmax, 10)
    x = np.log10(teff)
    for r in radii:
        l = get_L_from_r_teff(r, teff)
        y = np.log10(l)
        ax.plot(x, y, c="#808080", ls=":", lw=1)
        # ax.text(x[5], y[5], f"{r:.0f}"+r"$\,R_\odot$", fontsize=20,
        # transform=ax.transData, zorder=0, c="#808080"),
        # rotation=np.((max(y)-min(y))/(max(x)-min(x))))
    # reset ylim
    ax.set_ylim(ymin, ymax)


def get_BE_from_pfile(pfile, alpha_th=1.0, alpha_rot=0.0):
    """Calculates the binding energy profile of the star. See Eq. 6 in
    Dewi & Tauris 2000 (but change sign, binding energy>0 if layer is bound).

    The binding energy is calculated integrating the potential plus a
    fraction `alpha_th` of the internal energy.

    Parameters
    ----------
    pfile    : `MESA profile*.data` file
               assumed to contain the columns mass, energy, radius, and dm
    alpha_th : `float` optional, default 1
               fraction of thermal energy to include in the BE
    Returns
    -------
    BE       : `np.array` binding energy in cgs units
    """
    # from time import time
    # get the data in cgs units
    src, col = getSrcCol(pfile)
    m = src[:, col.index("mass")] * Msun  # g
    dm = src[:, col.index("dm")]  # g
    r = src[:, col.index("radius")] * Rsun_cm  # cm
    u = src[:, col.index("energy")]  # erg
    # calculate local gravitational potential
    psi = -1.0 * G_cgs * np.divide(m, r)  # erg
    if alpha_rot != 0:
        # calculate rotationa energy of spherical shell of
        # mass dm, outer radius r,
        # thickness dr, and rotation frequency omega
        omega = src[:, col.index("omega")]  # 1/sec
        I = (2.0 / 3.0) * np.square(r)  # specific moment of inertia cgs units
        erot = 0.5 * I * np.square(omega)
    else:
        erot = np.zeros(len(psi))
    # change sign: BE is the energy (>0) to provide to unbind the star
    BE = -1.0 * np.cumsum(np.multiply(psi + alpha_th * u + alpha_rot * erot, dm))
    return np.asarray(BE, dtype=float)


def plot_BE_r(
    pfile,
    ax,
    alpha_th=1.0,
    alpha_rot=0.0,
    top_axis=False,
    mark_he_core=False,
    **plot_kwargs,
):
    """
    plot the binding energy profile as a function of log(r/cm)
    """
    BE = get_BE_from_pfile(pfile, alpha_th=alpha_th, alpha_rot=alpha_rot)
    src, col = getSrcCol(pfile)
    logr = np.log10(src[:, col.index("radius")] * Rsun_cm)
    ax.plot(logr, BE, **plot_kwargs)
    if top_axis:
        """add top axis for mass"""
        ax.tick_params(axis="x", which="both", top=False)
        tx = ax.twiny()
        m = src[:, col.index("mass")]
        xmin, xmax = ax.get_xlim()
        imin = np.argmin(np.absolute(logr - xmin))
        imax = np.argmin(np.absolute(logr - xmax))
        tmin = m[imin]
        tmax = m[imax]
        tx.set_xlim(tmin, tmax)
        # tx.set_xticks([])
        # tx.set_xticklabels([])
        tx.set_xlabel(r"m $[M_\odot]$")
    if mark_he_core:
        m_he_core_01, m_he_core_1, m_he_core_2 = get_He_core_mass_from_pfile(pfile)
        m = src[:, col.index("mass")]
        i_01 = np.argmin(np.absolute(m - m_he_core_01))
        i_1 = np.argmin(np.absolute(m - m_he_core_1))
        i_2 = np.argmin(np.absolute(m - m_he_core_2))
        try:
            color = plot_kwargs["color"]
        except:
            color = plot_kwargs["c"]
        ax.axvline(logr[i_01], 0, 1, ls="-", lw=2, c=color, zorder=0)
        ax.axvline(logr[i_1], 0, 1, ls="--", lw=2, c=color, zorder=0)
        ax.axvline(logr[i_2], 0, 1, ls="-.", lw=2, c=color, zorder=0)


def plot_BE_m(pfile, ax, alpha_th=1.0, **plot_kwargs):
    """
    plot the binding energy profile as a function of mass
    """
    BE = get_BE_from_pfile(pfile, alpha_th=alpha_th)
    src, col = getSrcCol(pfile)
    # logr = np.log10(src[:, col.index("radius")] * Rsun_cm)
    m = src[:, col.index("mass")]
    ax.plot(m, BE, **plot_kwargs)


def get_ratio_BE(pfile1, pfile2, alpha_th=1.0, alpha_rot=0.0):
    """Ratio of pfile2/pfile1 on the x-coordinates in pfile1.

    Ratio of the binding energies (0<alpha_th<=1
    sets the fractional contribution of the internal energy -- 0<alpha_rot<=1
    sets the fractional contribution of the rotational energy) or
    gravitational energy (alpha_th=0, alpha_rot=0).

    Uses scipy.interpolate.interp1d to find a function that can then
    be evaluated on the other grid. The interpolation is done in
    fractional mass coordinate q=m/M.

    Parameters:
    -----------
    pfile1   :   `string` absolute path of MESA profile
    pfile2   :   `string` absolute path of MESA profile
    alpha_th :   `float`, optional fraction of thermal energy that is
                 considered in the binding energy. By default include the internal energy.
    alpha_rot:   `float`, optional fraction of rotational energy that is
                 considered in the binding energy. By default do not the rotational energy.

    Returns:
    -------
    ratio2_to_1: `np.array`, array of BE from pfile2/BE from pfile1 (or np.nan for the domain
                  of pfile2 outside the domain of pfile1 in mass)

    """
    # get data pfile1
    src1, col1 = getSrcCol(pfile1)
    BE1 = get_BE_from_pfile(pfile1, alpha_th=alpha_th, alpha_rot=alpha_rot)
    r1 = src1[:, col1.index("radius")]
    m1 = src1[:, col1.index("mass")]
    q1 = m1 / max(m1)
    # get data pfile2
    src2, col2 = getSrcCol(pfile2)
    BE2 = get_BE_from_pfile(pfile2, alpha_th=alpha_th, alpha_rot=alpha_rot)
    r2 = src2[:, col2.index("radius")]
    m2 = src2[:, col2.index("mass")]
    q2 = m2 / max(m2)
    # interpolate pfile2 on pfile1
    # MESA arrays are from the surface inwards, to have a monotonically increasing
    # mass coordinate, flip the arrays to interpolate
    x1 = q1[::-1]
    x2 = q2[::-1]
    y1 = BE1[::-1]
    y2 = BE2[::-1]
    interp2 = interp1d(
        x2, y2, assume_sorted=True, bounds_error=False, fill_value="extrapolate"
    )
    y2_interp = interp2(x1)
    ratio2_to_1 = y2_interp / y1
    # flip back to MESA convention (index 0 == surface)
    ratio2_to_1 = ratio2_to_1[::-1]
    return ratio2_to_1


def plot_ratio_BErot_BE_m(pfile, ax, **plot_kwargs):
    """
    For a MESA profile, plot the BE including rotational energy divided the BE
    not including rotation as a function of mass coordinate.


    Parameters:
    ----------
    pfile    :   `string` absolute path of MESA profile
    ax       :   `matplotlib.axes._subplots.AxesSubplot`

    Returns:
    --------
    ratio:       `np.array` ratio of BErot/BE
    """
    BE = get_BE_from_pfile(pfile, alpha_th=1.0, alpha_rot=0.0)
    BErot = get_BE_from_pfile(pfile, alpha_th=1.0, alpha_rot=1.0)
    ratio = BErot / BE
    print("min= ", min(ratio), "max=", max(ratio))
    src, col = getSrcCol(pfile)
    m = src[:, col.index("mass")]
    logr = np.log10(src[:, col.index("radius")] * Rsun_cm)
    ax.plot(logr, ratio, **plot_kwargs)
    ax.set_ylim(0.8, 1.2)
    return ratio


def plot_ratio_BE_r(pfile1, pfile2, ax, alpha_th=1.0, alpha_rot=0.0, **plot_kwargs):
    """
    Ratio of pfile2/pfile1 on the x-coordinates in pfile1.

    Ratio of the binding energies (0<alpha_th<=1
    sets the fractional contribution of the internal energy -- 0<alpha_rot<=1
    sets the fractional contribution of the rotational energy) or
    gravitational energy (alpha_th=0, alpha_rot=0).

    Uses scipy.interpolate.interp1d to find a function that can then
    be evaluated on the other grid. The interpolation is done in mass coordinate.
    Extrapolated values are set to nan (very center and very outermost layers typically).

    Parameters:
    -----------
    pfile1   :   `string` absolute path of MESA profile
    pfile2   :   `string` absolute path of MESA profile
    ax       :   `matplotlib.axes._subplots.AxesSubplot`
    alpha_th :   `float`, optional fraction of thermal energy that is
                considered in the binding energy. By default include the internal energy.
    alpha_rot:   `float`, optional fraction of rotational energy that is
                 considered in the binding energy. By default do not the rotational energy.


    Returns:
    -------
    ratio2_to_1: `np.array`, array of BE from pfile2/BE from pfile1 (or np.nan for the domain
                  of pfile2 outside the domain of pfile1 in mass)
    """
    src1, col1 = getSrcCol(pfile1)
    ratio2_to_1 = get_ratio_BE(pfile1, pfile2, alpha_th, alpha_rot)
    r1 = src1[:, col1.index("radius")]
    ax.plot(np.log10(r1 * Rsun_cm), ratio2_to_1, **plot_kwargs)
    return ratio2_to_1


def plot_ratio_BE_m(pfile1, pfile2, ax, alpha_th=1, **plot_kwargs):
    """
    Ratio of pfile2/pfile1 on the x-coordinates in pfile1.

    plots the ratio of the binding energies (0<alpha_th<=1
    sets the fractional contribution of the internal energy) or
    gravitational energy (alpha_th=0).

    Uses scipy.interpolate.interp1d to find a function that can then
    be evaluated on the other grid. The interpolation is done in mass coordinate.
    Extrapolated values are set to nan (very center and very outermost layers typically).

    Parameters:
    -----------
    pfile1  :   `string` absolute path of MESA profile
    pfile2  :   `string` absolute path of MESA profile
    ax      :   `matplotlib.axes._subplots.AxesSubplot`
    alpha_th:   `float`, optional fraction of thermal energy that is
                considered in the binding energy. By default include the internal energy.

    Returns:
    -------
    ratio2_to_1: `np.array`, array of BE from pfile2/BE from pfile1 (or np.nan for the domain
                  of pfile2 outside the domain of pfile1 in mass)
    """
    # get data pfile1
    src1, col1 = getSrcCol(pfile1)
    BE1 = get_BE_from_pfile(pfile1, alpha_th=alpha_th)
    m1 = src1[:, col1.index("mass")]
    # get data pfile2
    src2, col2 = getSrcCol(pfile2)
    BE2 = get_BE_from_pfile(pfile2, alpha_th=alpha_th)
    m2 = src2[:, col2.index("mass")]
    # interpolate pfile2 on pfile1
    # do the interpolation in mass coordinate and then plot in radius
    # MESA arrays are from the surface inwards, to have a monotonically increasing
    # mass coordinate, flip the arrays to interpolate
    x1 = m1[::-1]
    x2 = m2[::-1]
    y1 = BE1[::-1]
    y2 = BE2[::-1]
    # restrict the interpolation to the range of overlap
    xmin = max(min(x1), min(x2))
    xmax = min(max(x1), max(x2))
    ind = (x2 >= xmin) & (x2 <= xmax)
    interp2 = interp1d(x2[ind], y2[ind], assume_sorted=True, bounds_error=False)
    y2_interp = interp2(x1)
    ratio2_to_1 = y2_interp / y1
    ax.plot(m1, ratio2_to_1, **plot_kwargs)
    return ratio2_to_1


def get_ax_from_pfile(pfile, axes_list):
    """
    Parameters:
    ----------
    pfile : `string` absolute path to MESA profile file, must contain one of the strings below
    axes_list : `list` of mpl.axes objects
    Returns:
    -------
    one of the elements of the list
    """
    if "100Rsun" in pfile:
        return axes_list[0]
    elif "200Rsun" in pfile:
        return axes_list[1]
    elif "300Rsun" in pfile:
        return axes_list[2]
    elif "500Rsun" in pfile:
        return axes_list[3]
    elif "1000Rsun" in pfile:
        return axes_list[4]
    else:
        print(colored(pfile, "red"))
        raise ValueError(pfile + " not a known radius checkpoint")


def get_lambda_profile(pfile, alpha_th=1):
    """Calculates lambda_CE according to Eq. 1 of Dewi & Tauris 2000 or Eq. 9 of de Kool 1990
    (N.B.: Ivanova et al. 2013, fig. 5 y-axis label is 1/lambda not lambda)

    Parameters:
    --------
    pfile    : `string` path MESA profile*.data file
               assumed to contain the columns mass, energy, radius, and dm
    alpha_th : `float` optional
               fraction of thermal energy to include in the BE
    Returns
    -------
    L        : `np.array`, lambda values for each mesh point (same grid as input pfile)
    """
    # get the data in cgs units, load from center outwards
    src, col = getSrcCol(pfile)
    m = src[:, col.index("mass")]  # Msun
    M = max(m)
    # the radius is binary_separation*roche_lobe_radius, so fixed to the surface
    R = src[0, col.index("radius")]  # Rsun
    BE = get_BE_from_pfile(pfile, alpha_th=alpha_th)
    L = (G_cgs * M * (M - m) / R) / BE
    # add units back
    L *= Msun * Msun / Rsun_cm
    # print(f"{BE[i]:.5e}")
    # print(m[i]/Msun, (M-m[-1])/Msun, BE[-1])
    # print(L[i])
    # print(G_cgs)
    # print("-----------")
    return np.asarray(L, dtype=float)


def interpolate_BE_env(x1, x2, y2, offset=1.0):
    """
    Interpolates y2(x2) to get y2(x1)

    Parameters:
    ----------
    x1: `np.array`, independent coords we want
    y2, x2: `np.array`, need the same length
    offset : `float`, offset to make smaller numbers for interpolation

    Returns:
    --------
    y2_interp == y2(x1)
    """
    interpolator = interp1d(
        x2, y2 / offset, assume_sorted=True, bounds_error=False, fill_value=np.nan
    )
    y2_interp = interpolator(x1) * offset
    return np.array(y2_interp, dtype=float)


def plot_lambda_mass(pfile, ax, alpha_th=1, **plot_kwargs):
    l = get_lambda_profile(pfile, alpha_th=alpha_th)
    src, col = getSrcCol(pfile)
    m = src[:, col.index("mass")]
    ax.plot(m, l, **plot_kwargs)


def plot_lambda_r(pfile, ax, alpha_th=1, **plot_kwargs):
    l = get_lambda_profile(pfile, alpha_th=alpha_th)
    src, col = getSrcCol(pfile)
    r = src[:, col.index("radius")]
    ax.plot(r, l, **plot_kwargs)


def get_He_core_mass_from_pfile(pfile):
    """
    Returns the He core mass based on the default MESA
    definition
    """
    src, col = getSrcCol(pfile)
    x = src[:, col.index("h1")]
    y = src[:, col.index("he4")]
    m = src[:, col.index("mass")]
    Hecore = (y >= 0.1) & (x <= 0.01)
    m_he_core_01 = max(m[Hecore])
    Hecore = (y >= 0.1) & (x <= 0.1)
    m_he_core_1 = max(m[Hecore])
    Hecore = (y >= 0.1) & (x <= 0.2)
    m_he_core_2 = max(m[Hecore])
    return m_he_core_01, m_he_core_1, m_he_core_2


def plot_lambda_at_one_radius(
    ax, string, grid_folders, accretor=None, nonrot=None, plot_func=plot_lambda_mass
):
    """make plots of the lambda profiles comparing models

    Parameters:
    ----------
    ax :          `mpl.axes` where to plot
    string:       `str` profile name to be looked into the folders LOGS/LOGS2
    grid_folders: `str` path to grid of engineered models workdirectory
    accretor:     `str` or None, optional, path to the LOGS2 folder of the accretor
    nonrot:       `str` or None, optional, path to the work directory of a single star normal model
    plot_func  :  `python function`, can be `plot_lambda_mass` or
                  `plot_lambda_r` depending on if you want to plot
                  as a function of mass coordinate or radius
    legend:       `bool` whether to show the legend or not
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_folders)))

    # engineered stars
    for f in grid_folders:
        c = colors[grid_folders.index(f)]
        pfile = glob.glob(f + string)[0]
        plot_func(pfile, ax, alpha_th=1, c=c, lw=2, label=f.split("/")[-2])

    if accretor:
        pfile = glob.glob(accretor + "/" + string)[0]
        plot_func(pfile, ax, alpha_th=1, c="orange", label="accretor")
        m_he_core_01, m_he_core_1, m_he_core_2 = get_He_core_mass_from_pfile(pfile)
        ax.axvline(m_he_core_01, 0, 1, ls="-", lw=2, c="orange", zorder=0)
        ax.axvline(m_he_core_1, 0, 1, ls="--", lw=2, c="orange", zorder=0)
        ax.axvline(m_he_core_2, 0, 1, ls="-.", lw=2, c="orange", zorder=0)

    if nonrot:
        pfile = glob.glob(nonrot + string)[0]
        plot_func(pfile, ax, alpha_th=1, c="r", label="single")
        m_he_core_01, m_he_core_1, m_he_core_2 = get_He_core_mass_from_pfile(pfile)
        ax.axvline(m_he_core_01, 0, 1, ls="-", lw=2, c="red", zorder=0)
        ax.axvline(m_he_core_1, 0, 1, ls="--", lw=2, c="red", zorder=0)
        ax.axvline(m_he_core_2, 0, 1, ls="-.", lw=2, c="red", zorder=0)
