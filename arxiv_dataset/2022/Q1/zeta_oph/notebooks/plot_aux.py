# Authors:
#          Mathieu Renzo <mrenzo@flatironinstitute.org>

# Keywords: files

# Copyright (C) 2020-2021 Mathieu Renzo

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

__author__ = ['Mathieu Renzo <mrenzo@flatironinstitute.org']


import sys

# the line below import stuff from git@github.com:mathren90/plotFunc.git
sys.path.append("/mnt/home/mrenzo/codes/python_stuff/plotFunc/")
from MESAreader import getSrcCol
import numpy as np

# ---------------------------------------------------------------------------------------
# Zeta Ophiuchi's data
# Spectral type: O9.5Vnn
# xi=15 # no idea what this is, maybe microturbulence
def get_zeta_oph_inclination():
    """
    returns the inclination angle and error from Zehe et al. 2019
    """
    imin = 34  # degrees
    return imin


def get_zeta_oph_wind_mdot():
    """ data from marcolino et al. 2009"""
    log_mdot = -8.8
    err_log_mdot = 0.7
    return log_mdot, err_log_mdot


def get_zeta_oph_vsini():
    """
    returns the vsini and error from Zehe et al. 2019
    """
    vsini = 432  # km/s
    err_vsini = 16  # km/s
    return vsini, err_vsini


def zeta_oph_epsilon_he(ax):
    """
    plots the epsilon_he and error on ax
    data from Villamariz & Herrero 05
    """
    epsilon_he = 0.11  ## N(He/(N(H)+N(He)))
    err_epsilon_he_up = 0.05  # estimated from the range of values they explore
    err_epsilon_he_down = 0.02
    ax.errorbar(
        epslion_he, yerr=[err_epsilon_he_up, err_epsilon_he_down], fmt="o", color="r", zorder=1
    )


def zeta_oph_radius(ax):
    """
    plots the radius and error on ax
    data from Marcolino et al. 2009
    """
    R = 9.2  # Rsun
    err_Rp = 1.7
    err_Rm = 1.4
    age = 5  # Myr -- this is based on fits from single rotating models, not trustworthy
    ax.errorbar(age, R, yerr=[[err_Rm], [err_Rp]], fmt="o", color="r", zorder=1)


def get_zeta_oph_mass():
    """
    mass estimated by Villamariz & Herrero 2005:
    """
    # M = 19  # Msun
    # err_M = 11
    """
    mass estimated by Marcolino et al. 2009:
    """
    M = 13
    err_M_plus = 10
    err_M_minus = 7
    return M, [[err_M_minus], [err_M_plus]]


def zeta_oph_mass(ax):
    """
    plots the mass and error on ax
    data from Villamariz & Herrero 05
    """
    M, err_M = get_zeta_oph_mass()
    ax.errorbar(M, yerr=err_M, fmt="o", color="r", zorder=1)


def get_zeta_oph_logg():
    """
    data from Marcolino et al. 2009
    but Villamariz & Herrero 05 agrees too
    """
    logg = 3.6  # cm/s^2
    err_log_g = 0.2  # estimated from the range of values they explore
    return logg, err_log_g


def zeta_oph_spectroscopicHRD(ax):
    """
    plots Zeta ophiuchi logg and Teff on ax
    """
    logg, err_log_g = get_zeta_oph_logg()
    log_L, err_log_L, log_Teff, err_log_Teff = get_zeta_oph_L_teff()
    ax.errorbar(log_Teff, logg, xerr=err_log_Teff, yerr=err_log_g, fmt="o", color="r", zorder=1)


def get_zeta_oph_L_teff():
    """ returns luminosity and effective temperature
    from Marcolino et al. 2009 see their Tab. 3"""
    Teff = 32000  # K
    err_Teff = 2000
    # take log10
    log_Teff = np.log10(Teff)
    err_log_Teff = err_Teff / (Teff * np.log(10))
    # visual magnitude
    # Mv = -4.2 # taken from Howarth & Prinja 1989
    log_L = 4.86  # in Lsun units
    err_log_L = 0.1
    return log_L, err_log_L, log_Teff, err_log_Teff


def zeta_oph_HRD(ax):
    """
    plots Zeta ophiuchi L and Teff on ax
    """
    log_L, err_log_L, log_Teff, err_log_Teff = get_zeta_oph_L_teff()
    ax.errorbar(log_Teff, log_L, xerr=err_log_Teff, yerr=err_log_L, fmt="o", ms=15, color="r", zorder=10)

def get_zeta_oph_surface_he(X=0.7):
    """
    converts epsilon_he = N(He)/(N(H)+N(He)) with N number abundance
    to Y, mass fraction of He at the surface. Note that this takes X,
    the mass fraction of hydrogen as an optional argument. Data from
    Villamariz & Herrero 2005
    """
    epsilon = 0.11
    err_eps_p = 0.05  # errors estimated from the range shown in Tab. 1
    err_eps_m = 0.02
    # convert epsilon to Y
    # N.B: m_he4 = 4.002 atomic unit mass
    # and m_h1 = 1.007  atomic unit mass
    # so I'm going to assume m_he4 = 4m_h1 and neglect he3 here
    Y = 4 * X * epsilon / (1 - epsilon)
    err_Y_p = 4 * X * (1 / (1 - epsilon ** 2)) * err_eps_p
    err_Y_m = 4 * X * (1 / (1 - epsilon ** 2)) * err_eps_m
    return Y, err_Y_p, err_Y_m


def convert_eps_to_mass_frac(X_H, epsilon, err_epsilon, A_element):
    """
    converts epsilon = 12+log10(N_element/N_H) to mass_fraction
    A_element is the atomic mass. X_H the hydrogen mass fraction.
    """
    # Get the number ratio between the element and hydrogen
    N_element_div_N_H = 10.**(epsilon - 12.)
    # since:
    # N1/N2 = (X1/A1)/(X2/A2)
    # Therefore:
    # X1 = A1*(X2/A2)*(N1/N2)
    A_H = 1. # we neglect deuterium and tritium
    X_element = A_element*(X_H/A_H)*N_element_div_N_H
    err_X_element = A_element*(X_H/A_H)*N_element_div_N_H*np.log(10)*err_epsilon
    return X_element, err_X_element

def get_zeta_oph_surface_c(X=0.7):
    """
    converts epsilon = 12+log10(N_element/N_H) to mass_fraction
    Note the need for the hydrogen mass fraction as an input.
    Data from Villamariz & Herrero 2005.
    """
    epsilon_c = 7.86
    err_epsilon_c = 0.3
    A_c12 = 12.
    x_c, err_x_c = convert_eps_to_mass_frac(X, epsilon_c, err_epsilon_c, A_c12)
    return x_c, err_x_c


def get_zeta_oph_surface_n(X=0.7):
    """
    converts epsilon = 12+log10(N_element/N_H) to mass_fraction
    Note the need for the hydrogen mass fraction as an input.
    Data from Villamariz & Herrero 2005.
    """
    epsilon_n = 8.34
    err_epsilon_n = 0.3
    A_n14  = 14.
    x_n, err_x_n = convert_eps_to_mass_frac(X, epsilon_n, err_epsilon_n, A_n14)
    return x_n, err_x_n


def get_zeta_oph_surface_o(X=0.7):
    """
    converts epsilon = 12+log10(N_element/X) to mass_fraction
    Note the need for the hydrogen mass fraction as an input.
    Data from Villamariz & Herrero 2005.
    """
    epsilon_o = 8.69
    err_epsilon_o = 0.3
    A_o16 = 16.
    x_o, err_x_o = convert_eps_to_mass_frac(X, epsilon_o, err_epsilon_o, A_o16)
    return x_o, err_x_o


# -------------------------------------------------------------------
# radius plots
def get_radius_time(hfile):
    src, col = getSrcCol(hfile)
    t = src[:, col.index("star_age")] * 1e-6
    R = 10.0 ** (src[:, col.index("log_R")])
    return t, R


def plot_radius_time(ax, hfile1, c="#77CCCC", hfile2="", label=""):
    t, R = get_radius_time(hfile1)
    ax.plot(t, R, c=c, zorder=2, label=label)
    # post binary evolution part is optional
    if hfile2 != "":
        t, R = get_radius_time(hfile2)
        ax.plot(t, R, c=c, ls="-.", zorder=2, label=label)

# -------------------------------------------------------------------
# rotation
def get_surface_rotation_time(hfile):
    src, col = getSrcCol(hfile)
    t = src[:, col.index("star_age")] * 1e-6
    v = src[:, col.index("surf_avg_v_rot")]
    return t, v


def plot_surface_rotation_time(ax, hfile1, c="#77CCCC", hfile2="", label=""):
    t, v = get_surface_rotation_time(hfile1)
    ax.plot(t, v, c=c, zorder=2, label=label)
    # post binary evolution part is optional
    if hfile2 != "":
        t, v = get_surface_rotation_time(hfile2)
        ax.plot(t[0], v[0], c='k', zorder=2, marker='D', ms=10)
        ax.plot(t, v, c=c, zorder=2, label=label)

def get_omega_time(hfile):
    src, col = getSrcCol(hfile)
    omega_div_omega_crit = src[:, col.index("surf_avg_omega_div_omega_crit")]
    t = src[:, col.index("star_age")] * 1e-6  # in Myr
    return t, omega_div_omega_crit


def plot_omega_div_omega_crit(ax, hfile1, c="#77CCCC", ls="-", hfile2="", label=""):
    t, omega_div_omega_crit = get_omega_time(hfile1)
    ax.plot(t, omega_div_omega_crit, c=c, ls=ls, label=label, zorder=2)
    if hfile2 != "":
        t, omega_div_omega_crit = get_omega_time(hfile2)
        ax.plot(t, omega_div_omega_crit, c=c, ls="-.", zorder=2)


# -------------------------------------------------------------------
# Spectroscopic HRD
def getlogglogTeff(hfile):
    src, col = getSrcCol(hfile)
    log_Teff = src[:, col.index("log_Teff")]
    log_g = src[:, col.index("log_g")]
    return log_g, log_Teff


def plot_spectroscopic_HRD(ax, hfile1, c="#77CCCC", hfile2="", label=""):
    logg, logTeff = getlogglogTeff(hfile1)
    ax.plot(logTeff, logg, c=c, zorder=2, label=label)
    # post binary evolution part is optional
    if hfile2 != "":
        logg, logTeff = getlogglogTeff(hfile2)
        ax.plot(logTeff, logg, c=c, ls="-.", zorder=2)


# -------------------------------------------------------------------
# HRD
def getlogLlogTeff(hfile):
    src, col = getSrcCol(hfile)
    log_Teff = src[:, col.index("log_Teff")]
    logL = src[:, col.index("log_L")]
    return logL, log_Teff


def plot_HRD(ax, hfile1, c="#77CCCC", hfile2="", bfile="", label=""):
    logL, logTeff = getlogLlogTeff(hfile1)
    # if a binary history file is provided, print thicker lines during RLOF
    if bfile != "":
        src, col = getSrcCol(bfile)
        rl_relative_gap_1 = src[:, col.index("rl_relative_overflow_1")]
        ind = rl_relative_gap_1 > 0
        ax.plot(logTeff[ind], logL[ind], lw=8, c=c, zorder=2)
    # now make the plot
    ax.plot(logTeff, logL, c=c, zorder=2, label=label)
    # post binary evolution part is optional
    if hfile2 != "":
        logL, logTeff = getlogLlogTeff(hfile2)
        ax.plot(logTeff, logL, c=c, zorder=2)


# -------------------------------------------------------------------
# orbital velocity evolution
def plot_orbital_v2(ax, bfile, c="#77CCCC", label=""):
    srcb, colb = getSrcCol(bfile)
    v2 = srcb[:, colb.index("v_orb_2")]
    t = srcb[:, colb.index("age")] * 1e-6
    ax.plot(t, v2, lw=3, c=c, label=label)
    ax.text(t[-1] + 0.1, v2[-1], f"{v2[-1]:.1f}", fontsize=30)


# -------------------------------------------------------------------
# chemical composition
def get_epsilon_he(X, Y, Mtot=20, dq=1e-8):
    """
    convert mass fractions from MESA to the epslion spectroscopists like

    inputs:

    X, Y mass fractions of hydrogen and helium respectively,
    Mtot total mass
    dq fraction of the total mass that is the surface, 1e-8 is the MESA default

    output: epsilon_he = N(He)/(N(H)+N(He)) with N number abundance (?)

    TODO: verify this makes sense
    """
    # From $MESA_DIR/const/public/const_def.f90
    mp = 1.67262192369e-24  # grams
    mhe = 4 * mp  # neglects the binding energy of the he4 atom
    # get mass of the photosphere in grams
    mass = Mtot * Msun * dq
    # get number abundances from mass and mass fractions
    N_h = mass * X / mp
    N_he = mass * Y / mhe
    # calculate epsilon
    epsilon = N_he / (N_h + N_he)
    return epsilon


def get_epsilon(mass_frac, X):
    """
    convert mass fractions from MESA to the epslion spectroscopists like

    inputs:

    X mass fraction of hydrogen
    mass_frac mass fractions of the wanted element

    output: 12+log(mass_frac/X)

    TODO: verify this makes sense
    """
    return 12 + np.log10(mass_frac / X)


def plot_surface_abundances(hfile1, hfile2="", ax="", label="", legend=False, plot_expected=True):
    """
    plot the surface abundances of a few isotopes
    the post binary evolution is optional

    dash-dotted lines are the initial abundances.
    solid lines are the current abundances.
    dashed line are the conversion of the epsilon
    values reported in the literature for Zeta Ophiuchi. These depend
    on the amount of Y, so they have the same time dependence of the
    abundance of helium. Data from Villamariz & Herrero 2005.
    """
    src, col = getSrcCol(hfile1)

    he4_1 = src[:, col.index("surface_he4")]
    h1_1  = src[:, col.index("surface_h1")]
    n14_1 = src[:, col.index("surface_n14")]
    c12_1 = src[:, col.index("surface_c12")]
    o16_1 = src[:, col.index("surface_o16")]
    t_1   = src[:, col.index("star_age")] * 1e-6

    if hfile2 != "":
        src2, col2 = getSrcCol(hfile2)
        he4_2 = src2[:, col2.index("surface_he4")]
        h1_2  = src2[:, col2.index("surface_h1")]
        n14_2 = src2[:, col2.index("surface_n14")]
        c12_2 = src2[:, col2.index("surface_c12")]
        o16_2 = src2[:, col2.index("surface_o16")]
        t_2   = src2[:, col2.index("star_age")] * 1e-6

        he4 = np.concatenate((he4_1, he4_2))
        h1  = np.concatenate((h1_1, h1_2))
        n14 = np.concatenate((n14_1, n14_2))
        c12 = np.concatenate((c12_1, c12_2))
        o16 = np.concatenate((o16_1, o16_2))
        t   = np.concatenate((t_1, t_2))
    else:
        he4 = he4_1
        h1 = h1_1
        n14 = n14_1
        c12 = c12_1
        o16 = o16_1
        t = t_1

    if ax == "":
        # create plot figure if ax is not passed
        fig = plt.figure()
        gs = gridspec.GridSpec(100, 100)
        ax = fig.add_subplot(gs[:, :])

    if label != "":
        ax.set_title(label, fontsize=30)

    ax.plot(t, h1,  c="b", label=r"$^1\mathrm{H}$")
    ax.plot(t, he4, c="r", label=r"$^4\mathrm{He}$")
    ax.plot(t, c12, c="g", label=r"$^{12}\mathrm{C}$")
    ax.plot(t, n14, c="m", label=r"$^{14}\mathrm{N}$")
    ax.plot(t, o16, c="y", label=r"$^{16}\mathrm{O}$")

    ax.axhline(h1[0], 0, 1,  c="b", ls="-.", lw=1)
    ax.axhline(he4[0], 0, 1, c="r", ls="-.", lw=1)
    ax.axhline(n14[0], 0, 1, c="m", ls="-.", lw=1)
    ax.axhline(c12[0], 0, 1, c="g", ls="-.", lw=1)
    ax.axhline(o16[0], 0, 1, c="y", ls="-.", lw=1)

    ax.set_xlabel(r"$\mathrm{time \ [Myr]}$")
    ax.set_ylabel(r"$\mathrm{Surface\ mass\ fraction}\ X_i$")
    # ax.set_xlim(xmin=8.5, xmax=10)
    if plot_expected:
        xmin, xmax = ax.get_xlim()
        xcoord = np.linspace(xmin, xmax, len(h1))
        # helium
        Y, err_Y_p, err_Y_m = get_zeta_oph_surface_he(h1)
        ax.fill_between(xcoord, Y + err_Y_p, Y - err_Y_m, fc="r", alpha=0.2)
        ax.plot(xcoord, Y, c="r", ls="--", lw=2)

        # # oxygen
        Xo, err_Xo = get_zeta_oph_surface_o(h1)
        ax.fill_between(xcoord, Xo + err_Xo, Xo - err_Xo, fc="y", alpha=0.2)
        ax.plot(xcoord, Xo, c="y", ls="--", lw=2)

        # nitrogen
        Xn, err_Xn = get_zeta_oph_surface_n(h1)
        ax.fill_between(xcoord, Xn + err_Xn, Xn - err_Xn, fc="m", alpha=0.2)
        ax.plot(xcoord, Xn, c="m", ls="--", lw=2)

        # # carbon
        Xc, err_Xc = get_zeta_oph_surface_c(h1)
        ax.fill_between(xcoord, Xc + err_Xc, Xc - err_Xc, fc="g", alpha=0.2)
        ax.plot(xcoord, Xc, c="g", ls="--", lw=2)
    if legend:
        ax.legend(ncol=2, fontsize=20)
    # return the mass fractions at the end of the run
    return(h1[-1],
           he4[-1],
           n14[-1],
           c12[-1],
           o16[-1])

# ------------------------------------------------------------
# make Latex table with observed values
def make_table_zeta_Oph(outfname=""):
    if outfname == "":
        outfname = "/tmp/zeta_ophiuchi.text"
    with open(outfname, "w") as F:
        # write header
        outfname.writelines(
            [
                "\begin{table}",
                "\begin{center}",
                "\caption{Stellar parameters of $\zeta$ Ophiuchi.}",
                "\begin{tabular}{lc|c|c}",
                "\hline\hline",
                "Parameter & Units & Value & Ref.\\[2pt]",
            ]
        )
        # write main body
        vsini, errvisini = get_zeta_oph_vsini()
        line = (
            "$v\,\sin(i)$ & $\mathrm{km\ s^{-1}}$ & "
            + f"${vsini:.0f}"
            + "\pm"
            + f"{errvisini}"
            + "$ &"
            + " (1)\\"
        )
        outfname.writelines(line)
        # inclination
        min_i = get_zeta_oph_inclination()
        line = "$i$ & degrees & $\gtrim " + f"{min_i:.d}" + "$ & (1)\\"
        outfname.writelines(line)
        # L, Teff,
        log_L, err_log_L, log_Teff, err_log_Teff = get_zeta_oph_L_teff()
        line = "$\log_{10}(L/L_\odot)$ & &" + f"${log_L}\pm {err_log_L}$ & (2,3)\\"
        outfname.writelines(line)
        # logg
        logg, err_logg = get_zeta_oph_logg()
        line = "$\log_{10}(g/\mathrm{[cm\ s^{-2}]})$ & & " + f"${logg}\pm{err_logg}$" + " & (2)\\"
        outfname.writelines(line)
        # radius
        R, err_R = get_zeta_oph_R()
        line = "$R$ & $R_\odot$ & " + f"${R}\pm{err_R}$ & (2)\\"
        outfname.writelines(line)
        # mass
        M, err_M = get_zeta_oph_mass()
        line = "Mass & $M_\odot$ &" + f"${M}\pm{err_M}$& (2)\\"
        outfname.writelines(line)
        # surf composition

        # space velocity

        # write footer
        outfname.writelines(
            ["\hline", "\label{tab:star_param}", "\end{tabular}", "\end{center}", "\end{table}"]
        )


def MassVelocityEvolution(folder, convert=False, figName=""):
    """ folder is the binary evolution MESA folder """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print(folder)
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(200, 100)
    ax = fig.add_subplot(gs[:100, :])
    bx = fig.add_subplot(gs[100:200, :])
    bbx = bx.twinx()

    srcb, colb = getSrcCol(folder + "/binary_history.data", convert, convert)
    t = srcb[:, colb.index("age")] * 1e-6
    M1 = srcb[:, colb.index("star_1_mass")]
    M2 = srcb[:, colb.index("star_2_mass")]
    v2 = srcb[:, colb.index("v_orb_2")]
    P = srcb[:, colb.index("period_days")]

    ax.plot(t, M2, c="r", label=r"$M_2$")
    ax.plot(t, M1, c="b", label=r"$M_1$")
    ax.plot(t, M1 + M2, c="k", label=r"$M_1+M_2$")

    ax.text(t[-1] + 0.1, M2[-1], r"$M_2=" + f"{M2[-1]:.1f}" + r"$", fontsize=30)
    ax.text(t[-1] + 0.1, M1[-1], r"$M_1=" + f"{M1[-1]:.1f}" + r"$", fontsize=30)
    ax.text(t[-1] + 0.1, M1[-1] + M2[-1], r"$M_1+M_2=" + f"{M1[-1]+M2[-1]:.1f}" + r"$", fontsize=30)

    bx.plot(t, v2, ls="-", lw=3, c="r")
    bx.text(t[-1] + 0.1, v2[-1], f"{v2[-1]:.1f}", fontsize=30)
    bbx.plot(t, P, ls="--", lw=3, c="b")

    # ax.set_ylim(0,37)
    ax.set_xlim(0, 12)
    ax.set_xticklabels([])
    bx.set_xlim(ax.get_xlim())
    # bbx.set_ylim(90,700)
    bx.set_xlabel(r"$\mathrm{t \ [Myr]}$")
    ax.set_ylabel(r"$M \ [M_\odot]$")
    bx.set_ylabel(r"$v_2 \ [\mathrm{km\ s^{-1}}]$")
    bbx.set_ylabel(r"$P \ \mathrm{[days]}$", color="b")

    if figName != "":
        plt.savefig(figName)


#


def get_age_from_profile(pfile):
    # print(pfile)
    with open(pfile, 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                header_cols = line.split()
                # print(header_cols)
            if i == 2:
                header_data = line.split()
                break
    age = float(header_data[header_cols.index('star_age')])*1e-6
    return age # in Myr


def get_modnum_from_profile(pfile):
    # print(pfile)
    with open(pfile, 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                header_cols = line.split()
                # print(header_cols)
            if i == 2:
                header_data = line.split()
                break
    mn = int(header_data[header_cols.index('model_number')])
    return mn # in yr

def get_ZAMS_abundances(hfile):
    src, col = getSrcCol(hfile)
    surface_he4 = src[0, col.index("surface_he4")]
    surface_c12 = src[0, col.index("surface_c12")]
    surface_n14 = src[0, col.index("surface_n14")]
    surface_o16 = src[0, col.index("surface_o16")]
    return surface_he4, surface_c12, surface_n14, surface_o16


def get_profile_from_modnum(num, LOGS):
    """ returns the closest profile number to the model number wanted """
    index = LOGS+'/profiles.index'
    src = np.genfromtxt(index, skip_header=1)
    profile_num = src[:, -1]
    model_num = src[:, 0]
    i = np.argmin(np.absolute(model_num-num))
    return LOGS+'profile'+str(int(profile_num[i]))+'.data'


def get_center_h1(pfile):
    """ returns central value of h1 mass fraction of a profile"""
    src, col = getSrcCol(pfile)
    h1_center = src[-1, col.index("h1")]
    return h1_center



def plot_Dmix(pfile, ax="", legend=False):
    if ax == "":
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(100, 100)
        ax = fig.add_subplot(gs[:, :])
    age = get_age_from_profile(pfile)
    timestamp = f"${age:.3f}"+r"\,\mathrm{Myr}$"

    src, col = getSrcCol(pfile)
    mass = src[:, col.index("mass")]
    D_mix = 10**(src[:, col.index('log_D_mix')])
    D_mix_non_rotation = 10.0**(src[:, col.index('log_D_mix_non_rotation')])
    D_conv = 10.0**(src[:, col.index('log_D_conv')])
    D_semi = 10.0**(src[:, col.index('log_D_semi')])
    D_thrm = 10.0**(src[:, col.index('log_D_thrm')])

    D_ov = D_mix_non_rotation - D_thrm - D_semi
    D_rot = D_mix - D_mix_non_rotation

    # ax.scatter(mass, D_mix, lw=10, label="$\mathrm{Total}$", c="c")
    ax.scatter(mass, D_rot, lw=0, s=5, label="$\mathrm{Rotation}$", c='#FFCE54')
    ax.fill_between(mass, D_rot, alpha=0.75, fc='#FFCE54')

    ax.scatter(mass, D_ov, lw=0, s=5, label="$\mathrm{Overshooting}$", c='b')
    ax.fill_between(mass, D_ov, alpha=0.75, fc='b')

    ax.scatter(mass, D_conv, lw=0, s=5, label="$\mathrm{Convection}$", c="r")
    ax.fill_between(mass, D_conv, alpha=0.75, fc='r')

    # ax.scatter(mass, D_semi, lw=0, s=5, label="$\mathrm{Semiconvection}$", c="#DDDD77")
    # ax.fill_between(mass, D_semi, alpha=0.75, fc='#DDDD77')

    ax.scatter(mass, D_thrm, lw=0, s=5, label="$\mathrm{Thermohaline}$", c="#CC99BB")
    ax.fill_between(mass, D_thrm, alpha=0.75, fc='#CC99BB')

    ax.text(0.05,0.1, timestamp, fontsize=30, va="center",
            transform=ax.transAxes, bbox=dict(facecolor='w',
                                              edgecolor='black', boxstyle='round,pad=0.2', alpha=0.9))
    ax.set_yscale('log')
    if legend:
        conv, = ax.plot(np.nan, np.nan, lw=30, ls='-', c='r',label="$\mathrm{Convection}$")
        ov, = ax.plot(np.nan, np.nan, lw=30, ls='-', c='b',label="$\mathrm{Overshooting}$")
        rot, = ax.plot(np.nan, np.nan, lw=30, ls='-', c='#FFCE54',label="$\mathrm{Rotation}$")
        thrm, = ax.plot(np.nan, np.nan, lw=30, ls='-', c='#CC99BB',label="$\mathrm{Thermohaline}$")
        # semiconv, = ax.plot(np.nan, np.nan, lw=30, ls='-', c='#DDDD77',label="$\mathrm{Semiconvection}$")
        ax.legend(handles=[conv, ov, rot, thrm], loc='lower left', bbox_to_anchor= (0, 1.01), ncol=2,
            borderaxespad=0, frameon=False, handlelength=0.5)

def get_center_h1(pfile):
    src, col = getSrcCol(pfile)
    h1_center = src[-1, col.index("h1")]
    return h1_center



def get_three_colormaps(z1, z2, z3):
    import matplotlib as mpl
    """
    returns three smoothly transitioning colormaps
    covering the intervals [0, z1], [z1, z2], and [z2, z3]

    parameters: z1, z2, z3 that define the edges of three color regions

    returns three colormaps with their vmin, vmax, and normalizations
    """
    # colors that work
    c1 = 'r'
    c2 = 'gold'
    c3 = 'c'
    c4 = 'fuchsia'

    cmap1 =  mpl.colors.LinearSegmentedColormap.from_list('cmap1', [c1,c2])
    vmin1 = 0
    vmax1 = z1
    Ncolors1 = 10
    bounds1 = np.linspace(vmin1, vmax1, Ncolors1)
    norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)

    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('cmap2', [c2, c3])
    vmin2 = vmax1
    vmax2 = z2
    Ncolors2 = 10
    bounds2 = np.linspace(vmin2, vmax2, Ncolors2)
    norm2 = mpl.colors.BoundaryNorm(bounds2, cmap2.N)

    cmap3 = mpl.colors.LinearSegmentedColormap.from_list('cmap3', [c3, c4])
    vmin3 = vmax2
    vmax3 = z3
    Ncolors3 = 10
    bounds3 = np.linspace(vmin3, vmax3, Ncolors3)
    norm3 = mpl.colors.BoundaryNorm(bounds3, cmap3.N)

    return(cmap1, vmin1, vmax1, norm1, cmap2, vmin2, vmax2, norm2, cmap3, vmin3, vmax3, norm3)
