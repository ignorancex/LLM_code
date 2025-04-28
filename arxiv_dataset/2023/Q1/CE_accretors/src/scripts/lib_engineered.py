# Author:
#          Mathieu Renzo <mrenzo@flatironinstitute.org>
#
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

import sys

# import socket
from MESAreader import getSrcCol, Rsun_cm, Lsun
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from termcolor import colored
except:
    from MESAreader import colored
import paths

# ---------------------------------------
def get_xq(m, Mtot):
    """given a lagrangian mass coordinate
    calculates the fraction of Mtot above it"""
    xq = (Mtot - m) / Mtot
    return xq


def mk_line(x0, y0, xx, y1):
    return y0 + ((y1 - y0) / (max(xx) - x0)) * (xx - x0)


def get_M_boundary(pfile, h1_outer=None, h1_inner=None, offset=1e-2, plot=False):
    """calculate mass of the He core-envelope boundary region based on the h1 and he4 profiles.

    This encloses the H-burning shell, (if any) and corresponds to the region from
    where H1 goes from the central to the surface value.

    Parameters:
    -----------
    pfile : `string` path to profile*.data h1_outer: `float` outer value of hydrogen abundance, defaults to surface h1_inner: `float` inner value of hydrogen abundance, defaluts to core offset: `float` offset from inner and outer value plot: `bool` makes a plot
    h1_outer: `float`, optional, value of H mass fraction at surface, if None, takes outermost cell
    h1_inner: `float`, optional, value of H mass fraction in the core, if None, takes innermost cell
    offset: `float`, optional, offset from values above defining the core-envelope boundary region.
    plot: `bool`, optional, create a figure

    Returns:
    --------
    delta_M_boundary: `float` Thickness of transition region
    max_M_boundary: `float` max ...
    min_M_boundary: `float` min ...
    """

    src, col = getSrcCol(pfile)
    h1 = src[:, col.index("h1")]
    he3 = src[:, col.index("he3")]
    he4 = src[:, col.index("he4")]
    m = src[:, col.index("mass")]
    r = src[:, col.index("radius")] * Rsun_cm

    if not h1_outer:
        h1_outer = h1[0]
    if not h1_inner:
        h1_inner = h1[-1]
    # index slicing for core-envelope boundary region
    ind = (h1 <= h1_outer - offset) & (h1 > h1_inner + offset)
    if plot:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(100, 100)
        ax = fig.add_subplot(gs[:, :])
        text = pfile.split("/")[-1].split(".data")[0]
        title = text.replace("Rsun", r" $R_\odot$")
        ax.set_title(title, fontsize=30)
        ax.plot(m, h1, label="h1")
        ax.plot(m, he4, label="he4")
        ax.fill_between((min(m[ind]), max(m[ind])), 0, 1, zorder=0, alpha=0.5)
        ax.set_xlabel(r"m $M_\odot$")
        ax.set_ylabel(r"$X_i$")
        ax.legend()
        folder = get_folder_from_file(pfile)
        if "index" in folder:
            # binary run
            m1, m2 = get_masses(folder)
            text = "accretor_" + f"{m1:.0f}_" + text
        else:
            # single star run
            text = folder + "_" + text
        print(colored("/" + text + "M_boundary.png", "blue"))
        plt.savefig("/" + text + "M_boundary.png")
        plt.close()
    delta_M_boundary = max(m[ind]) - min(m[ind])
    max_M_boundary = max(m[ind])
    min_M_boundary = min(m[ind])
    return delta_M_boundary, max_M_boundary, min_M_boundary


def mk_simplified_profile_from_pfile(
    pfile, outfile=None, plot=False, m_in=None, m_out=None
):
    """takes surface and hydrogen mass fraction and entropy, and thickness
    of the core/envelope boundary region and produces a simple entropy and
    composition profile to initialize MESA. Assume the Z abundance are not
    affected. The input model should be a TAMS model or close to it.

    Assumes the models are computed with `approx21_plus_cr56.net` which
    has the following isotopes:

     1   neut
     2   h1
     3   prot
     4   he3
     5   he4
     6   c12
     7   n14
     8   o16
     9   ne20
    10   mg24
    11   si28
    12   s32
    13   ar36
    14   ca40
    15   ti44
    16   cr48
    17   cr56
    18   fe52
    19   fe54
    20   fe56
    21   co56
    22   ni56

    Parameters:
    -----------
    pfile  : `string` path to input profile*.data
    outfile: `string`, optional, path to where to save the initial conditions for MESA
    plot   : `bool`, optional, make a plot
    m_in   : `float` optional, if provided will be used instead of the min_M_boundary from get_M_boundary
    m_out  : `float` optional, if provided will be used instead of the max_M_boundary get_M_boundary values

    Returns:
    -------
    num_points     : `int` number of mesh points for the simple model
    xq             : `np.array` fraction of mass external to the coordinate considered
    s_out          : `np.array` specific entropy in units of erg/g/K
    h1_out, he4_out: `np.array`s mass fraction arrays for h1 and he4

    """
    # check if call makes sense
    if ((m_in != None) and (m_out == None)) or ((m_in == None) and (m_out != None)):
        raise ValueError("provide both or neither m_in and m_out!")
    # get data
    src, col = getSrcCol(pfile)
    mass = src[:, col.index("mass")]
    entropy = src[:, col.index("entropy")]
    # composition
    neut = src[:, col.index("neut")]
    h1 = src[:, col.index("h1")]
    prot = src[:, col.index("prot")]
    he3 = src[:, col.index("he3")]
    he4 = src[:, col.index("he4")]
    c12 = src[:, col.index("c12")]
    n14 = src[:, col.index("n14")]
    o16 = src[:, col.index("o16")]
    ne20 = src[:, col.index("ne20")]
    mg24 = src[:, col.index("mg24")]
    si28 = src[:, col.index("si28")]
    s32 = src[:, col.index("s32")]
    ar36 = src[:, col.index("ar36")]
    ca40 = src[:, col.index("ca40")]
    ti44 = src[:, col.index("ti44")]
    cr48 = src[:, col.index("cr48")]
    cr56 = src[:, col.index("cr56")]
    fe52 = src[:, col.index("fe52")]
    fe54 = src[:, col.index("fe54")]
    fe56 = src[:, col.index("fe56")]
    co56 = src[:, col.index("co56")]
    ni56 = src[:, col.index("ni56")]
    ## plot input model
    if plot:
        fig, ax = plt.subplots()
        ax.plot(mass, h1, "b", label="h1 input")
        ax.plot(mass, he4, "g", label="he4 input")
        # ax.set_yscale('log')
        # ax.set_ylim(1e-4,1)
        bx = ax.twinx()
        bx.plot(mass, entropy, "r", label="s input")
    # build output
    num_points = len(mass)
    xq = get_xq(mass, max(mass))
    s_out = np.zeros(len(mass))
    h1_out = np.zeros(len(mass))
    he4_out = np.zeros(len(mass))
    # get extremal values
    (
        original_delta_M_boundary,
        original_max_M_boundary,
        original_min_M_boundary,
    ) = get_M_boundary(pfile)
    if (m_in == None) and (m_out == None):
        # print(colored("no m_in, m_out, let me calculate them from "+pfile, "blue"))
        min_M_boundary = original_min_M_boundary
        max_M_boundary = original_max_M_boundary
    else:
        min_M_boundary = m_in
        max_M_boundary = m_out
    i_out = np.argmin(np.absolute(mass - max_M_boundary))
    i_in = np.argmin(np.absolute(mass - min_M_boundary))
    if (max_M_boundary - min_M_boundary) >= original_delta_M_boundary:
        # widening the transition region
        h_out = h1[i_out]
        h_in = h1[i_in]
        he_out = he4[i_out]
        he_in = he4[i_in]
        entropy_out = entropy[i_out]
        entropy_in = entropy[i_in]
        # split domain in regions core/transition/envelope
        index_in = mass <= min_M_boundary
        index_boundary = (mass <= max_M_boundary) & (mass >= min_M_boundary)
        index_out = mass >= max_M_boundary
        ## # create output arrays
        # entropy
        s_out[index_in] = entropy[index_in]
        s_out[index_boundary] = mk_line(
            min(mass[index_boundary]), entropy_in, mass[index_boundary], entropy_out
        )
        s_out[index_out] = entropy[index_out]
        if plot:
            bx.scatter(mass, s_out, color="orange", label="s output")
        # composition
        h1_out[index_in] = h1[index_in]
        h1_out[index_boundary] = mk_line(
            min(mass[index_boundary]), h_in, mass[index_boundary], h_out
        )
        h1_out[index_out] = h1[index_out]
        if plot:
            ax.scatter(mass, h1_out, c="c", label="h1 output")
        he4_out[index_in] = he4[index_in]
        he4_out[index_boundary] = mk_line(
            min(mass[index_boundary]), he_in, mass[index_boundary], he_out
        )
        he4_out[index_out] = he4[index_out]
        if plot:
            ax.scatter(mass, he4_out, c="m", label="he4 output")
            ax.legend(fontsize=20, loc="center left")
    else:
        # shrinking: we have to extrapolate from the edges of the
        # original_delta_M_boundary to the mass coordinate wanted
        original_i_out = np.argmin(np.absolute(mass - original_max_M_boundary))
        original_i_in = np.argmin(np.absolute(mass - original_min_M_boundary))
        h_out = h1[original_i_out]
        h_in = h1[original_i_in]
        he_out = he4[original_i_out]
        he_in = he4[original_i_in]
        entropy_out = entropy[original_i_out]
        entropy_in = entropy[original_i_in]
        # ordering original_min_M_boundary< min_M_boundary<  max_M_boundary<  original_max_M_boundary
        index_in = mass <= original_min_M_boundary
        index_layer_in = (mass > original_min_M_boundary) & (mass <= min_M_boundary)
        index_intermediate = (mass > min_M_boundary) & (mass <= max_M_boundary)
        index_layer_out = (mass > max_M_boundary) & (mass <= original_max_M_boundary)
        index_out = mass >= original_max_M_boundary
        # entropy
        s_out[index_in] = entropy[index_in]
        s_out[index_layer_in] = entropy_in
        s_out[index_intermediate] = mk_line(
            min(mass[index_intermediate]),
            entropy_in,
            mass[index_intermediate],
            entropy_out,
        )
        s_out[index_layer_out] = entropy_out
        s_out[index_out] = entropy[index_out]
        if plot:
            bx.scatter(mass, s_out, color="orange", label="s output")
        # composition
        h1_out[index_in] = h1[index_in]
        h1_out[index_layer_in] = h_in
        h1_out[index_intermediate] = mk_line(
            min(mass[index_intermediate]), h_in, mass[index_intermediate], h_out
        )
        h1_out[index_layer_out] = h_out
        h1_out[index_out] = h1[index_out]
        if plot:
            ax.scatter(mass, h1_out, c="c", label="h1 output")
        he4_out[index_in] = he4[index_in]
        he4_out[index_layer_in] = he_in
        he4_out[index_intermediate] = mk_line(
            min(mass[index_intermediate]), he_in, mass[index_intermediate], he_out
        )
        he4_out[index_layer_out] = he_out
        he4_out[index_out] = he4[index_out]
    if outfile:
        # Make composition simplified input profile
        with open(outfile + "_composition.txt", "w") as f:
            f.writelines(
                str(len(mass)) + " " + str(22) + "\n"
            )  #   1st line: num_points num_species
            for i in range(len(mass)):
                line = (
                    f"{xq[i]:.16f}"
                    + "    "
                    + f"{neut[i]:.16f}"
                    + "    "
                    + f"{h1_out[i]:.16f}"
                    + "    "
                    + f"{prot[i]:.16f}"
                    + "    "
                    + f"{he3[i]:.16f}"
                    + "    "
                    + f"{he4_out[i]:.16f}"
                    + "    "
                    + f"{c12[i]:.16f}"
                    + "    "
                    + f"{n14[i]:.16f}"
                    + "    "
                    + f"{o16[i]:.16f}"
                    + "    "
                    + f"{ne20[i]:.16f}"
                    + "    "
                    + f"{mg24[i]:.16f}"
                    + "    "
                    + f"{si28[i]:.16f}"
                    + "    "
                    + f"{s32[i]:.16f}"
                    + "    "
                    + f"{ar36[i]:.16f}"
                    + "    "
                    + f"{ca40[i]:.16f}"
                    + "    "
                    + f"{ti44[i]:.16f}"
                    + "    "
                    + f"{cr48[i]:.16f}"
                    + "    "
                    + f"{cr56[i]:.16f}"
                    + "    "
                    + f"{fe52[i]:.16f}"
                    + "    "
                    + f"{fe54[i]:.16f}"
                    + "    "
                    + f"{fe56[i]:.16f}"
                    + "    "
                    + f"{co56[i]:.16f}"
                    + "    "
                    + f"{ni56[i]}"
                    + "\n"
                )
                f.writelines(line)
        # Make entropy simplified input profile
        with open(outfile + "_entropy.txt", "w") as f:
            f.writelines(str(len(mass)) + "\n")  #   1st line: num_points
            for i in range(len(mass)):
                line = f"{xq[i]:.16f}" + "    " + f"{s_out[i]:.16f}" + "\n"
                f.writelines(line)
    return (num_points, xq, s_out, h1_out, he_out)


def plot_entropy(pfile, ax, **plot_kwargs):
    src, col = getSrcCol(pfile)
    m = src[:, col.index("mass")]
    s = src[:, col.index("entropy")]
    ax.plot(m, s, **plot_kwargs)


def plot_XY(pfile, ax, **plot_kwargs):
    src, col = getSrcCol(pfile)
    m = src[:, col.index("mass")]
    h1 = src[:, col.index("h1")]
    he4 = src[:, col.index("he4")]
    # overwrite linestyle
    plot_kwargs["linestyle"] = "-"
    ax.plot(m, h1, **plot_kwargs)
    plot_kwargs["linestyle"] = "--"
    ax.plot(m, he4, **plot_kwargs)


def plot_s_h_he(init_model, grid_folders, accretor, ax_top, ax_bottom):
    """helper function for fig. 5"""

    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_folders)))

    delta_M_bound, M_bound_min, M_bound_max = get_M_boundary(init_model)
    plot_entropy(
        init_model,
        ax_top,
        lw=3,
        ls="-",
        color="r",
        label=init_model.split("/")[-3].split("_rot")[0] + "$M_\odot$",
        zorder=10,
    )
    plot_XY(init_model, ax_bottom, lw=3, c="r", zorder=10)

    for f in grid_folders:
        pfile = f + "profile1.data"
        label = ""  # f.split('/')[-2]
        delta_M_bound, M_bound_min, M_bound_max = get_M_boundary(pfile, offset=0.05)
        ax_top.axvspan(M_bound_min, M_bound_max, fc="#808080", alpha=0.1, zorder=0)
        ax_bottom.axvspan(M_bound_min, M_bound_max, fc="#808080", alpha=0.1, zorder=0)
        # print(f, M_bound_min, M_bound_max)
        # label = f"{delta_M_bound:.3f}" # f.split('/')[-2]
        c = colors[grid_folders.index(f)]
        plot_entropy(pfile, ax_top, c=c, lw=2, label=label)
        plot_XY(pfile, ax_bottom, c=c, lw=2, label="")

    # add accretor
    plot_entropy(
        accretor + "/recomputed_TAMS.data",
        ax_top,
        lw=3,
        ls="-",
        color="orange",
        label="accretor",
        zorder=10,
    )
    plot_XY(accretor + "/recomputed_TAMS.data", ax_bottom, lw=3, c="orange", zorder=10)
    ax_bottom.set_xlim(ax_top.get_xlim())
    ax_bottom.set_xlabel(r"$m \ [M_\odot]$")
    ax_top.set_xticklabels([])
    ax_top.set_xlabel("")


def three_panel_plot_s_h_he(
    grid_folders1,
    init_model1,
    accretor1,
    nonrot1,
    grid_folders2,
    init_model2,
    accretor2,
    nonrot2,
    grid_folders3,
    init_model3,
    accretor3,
    nonrot3,
    fig_name=None,
):
    """helper fuction for plotting"""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(100, 120)
    gs.update(wspace=0, hspace=0)  # top=1.1)
    ax1 = fig.add_subplot(gs[:50, :40])
    bx1 = fig.add_subplot(gs[50:, :40])
    ax2 = fig.add_subplot(gs[:50, 40:80])
    bx2 = fig.add_subplot(gs[50:, 40:80])
    ax3 = fig.add_subplot(gs[:50, 80:])
    bx3 = fig.add_subplot(gs[50:, 80:])

    plot_s_h_he(init_model1, grid_folders1, accretor1, ax1, bx1)
    plot_s_h_he(init_model2, grid_folders2, accretor2, ax2, bx2)
    plot_s_h_he(init_model3, grid_folders3, accretor3, ax3, bx3)

    bx1.set_ylabel(r"$X_i$")
    ax1.set_ylabel("s \ [k_{B}N_{A}]}")
    ax2.set_yticklabels([])
    bx2.set_yticklabels([])
    ax3.set_yticklabels([])
    bx3.set_yticklabels([])
    # ylim
    ax2.set_ylim(ax1.get_ylim())
    ax3.set_ylim(ax1.get_ylim())
    bx2.set_ylim(bx1.get_ylim())
    bx3.set_ylim(bx1.get_ylim())
    # xlim
    # ax1.set_xlim(bx1.get_xlim())
    # ax2.set_xlim(bx2.get_xlim())
    # ax3.set_xlim(bx3.get_xlim())

    # legend
    bx1.plot(np.nan, np.nan, ls="-", c="k", label=r"$^1\mathrm{H}$")
    bx1.plot(np.nan, np.nan, ls="--", c="k", label=r"$^4\mathrm{He}$")
    bx1.legend(
        handletextpad=0.4, handlelength=0.75, columnspacing=0.75, loc="center left"
    )

    ax1.legend(handletextpad=0.5, frameon=True)
    ax2.legend(handletextpad=0.5, frameon=True)
    ax3.legend(handletextpad=0.5, frameon=True)
    fig.align_labels()
    if fig_name:
        plt.savefig(fig_name)


def get_dm_from_pfile_eng(pfile):
    """
    get the `dm` by which we shift the upper-edge of the CEB of a
    normal single star to make the engineered models. The dm is read
    from the folder name

    see also /src/data/MESA_input/grid_management_scripts/setup_engineered.py

    Parameters:
    ----------
    `pfile`: `str` or path of the engineered star,

    Returns:
    -------
    `dm`:   `float` (can be <0 for shifts towards the center)
    """
    return float(pfile.split("/")[-3])


def sorter_engineered_profiles(pfile):
    # get corresponding normal single star model
    mass = pfile.split("/")[-4].lstrip("grid")
    init_model = (
        str(paths.data)
        + "/MESA_output/engineered_stars/TAMS_models/"
        + str(mass)
        + "_rot0_to_TAMS/LOGS/TAMS.data"
    )
    # get outer CEB boundary in the reference TAMS model
    delta_M_bound, M_bound_max, M_bound_min = get_M_boundary(init_model)
    # get shift from reference TAMS model
    dm = get_dm_from_pfile_eng(pfile)  # can be <0
    # return final outer boundary of the engineered model
    return M_bound_max + dm
