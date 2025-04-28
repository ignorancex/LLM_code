"""Quantifying mass loss in observations"""
import cmasher as cmr
from icecream import ic
from matplotlib import cm, colors as mplcolors, patheffects as pe, pyplot as plt, ticker
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binstat, binned_statistic_2d as binstat2d
import warnings

from plottery.plotutils import update_rcParams

update_rcParams()

from hbtpy import hbt_tools
from hbtpy.hbt_tools import load_subhalos, save_plot
from hbtpy.subhalo import Subhalos
from hbtpy.helpers.plot_auxiliaries import get_axlabel, get_bins
from hbtpy.helpers.plot_definitions import axlabel, xbins
from hbtpy.helpers.plot_relations import apply_bins


warnings.simplefilter("once", RuntimeWarning)


def main():
    args = hbt_tools.parse_args()
    isnap = -1
    sim, subs = load_subhalos(args, isnap=isnap)
    subs = Subhalos(
        subs,
        sim,
        isnap,
        load_velocities=False,
        logMstar_min=9,
        logMmin=None,
        logM200Mean_min=None,
    )
    s = subs.satellite_mask

    # min masses are the same as in mass_relations.py
    mstarbins = np.log10(xbins["Mstar"])
    x = 10 ** ((mstarbins[1:] + mstarbins[:-1]) / 2)
    mstarbins = 10**mstarbins

    h = "history:max_Mbound"
    bincol = f"{h}:time"
    bincol = "M200Mean"
    bincol = "ComovingMostBoundDistance01/R200MeanComoving"
    xcol = "Mstar"
    logbins = "time" not in bincol
    bins = get_bins(bincol, logbins=logbins, n=4)

    cmap_range = (0.1, 0.8)
    cmap = cmr.get_sub_cmap("viridis", *cmap_range)
    (
        mask,
        bins,
        bin_centers,
        bindata,
        binlabel,
        vmin,
        vmax,
        colormap,
        colors,
    ) = apply_bins(subs, s, bincol, bins, "mean", logbins, None, cmap)
    cmap = plt.get_cmap(cmap, bin_centers.size)
    boundary_norm = mplcolors.BoundaryNorm(bins, cmap.N)
    sm = cm.ScalarMappable(cmap=cmap, norm=boundary_norm)
    sm.set_array([])
    ic(colors)

    # fig = plt.figure(figsize=(8,7), constrained_layout=True)
    # gs = GridSpec(3, 1, height_biass=(1,1,1), hspace=0.05,
    #               left=0.15, right=0.9, bottom=0.1, top=0.95)
    # fig.add_subplot(gs[0])
    # fig.add_subplot(gs[1])
    # fig.add_subplot(gs[2])
    # axes = fig.axes
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

    lw = 3
    border = [pe.Stroke(linewidth=lw + 1, foreground="w"), pe.Normal()]
    kwargs_fill = dict(alpha=0.5, lw=0, zorder=-1)
    args_all = ("k-",)
    kwargs_all = dict(lw=lw, zorder=-9)  # , path_effects=border)
    kwargs_fill_all = dict(color="0.7", lw=0, zorder=-10)

    ax = axes[0]
    ycol = f"Mbound/{h}:Mbound"
    # these are actually just mass ratios
    massloss = binstat2d(
        subs[xcol][s], subs[bincol][s], subs[ycol][s], "mean", (mstarbins, bins)
    ).statistic.T
    massloss_lo = binstat2d(
        subs[xcol][s], subs[bincol][s], subs[ycol][s], error_lo, (mstarbins, bins)
    ).statistic.T
    massloss_hi = binstat2d(
        subs[xcol][s], subs[bincol][s], subs[ycol][s], error_hi, (mstarbins, bins)
    ).statistic.T
    # this is really mass loss
    massloss = 1 - massloss
    massloss_lo, massloss_hi = massloss_hi, massloss_lo
    ic(massloss)
    for i, (m, mlo, mhi, c) in enumerate(
        zip(massloss, massloss_lo, massloss_hi, colors)
    ):
        ax.fill_between(x, m - mlo, m + mhi, color=c, **kwargs_fill)
        ax.plot(x, m, "-", color=c, lw=lw, zorder=2)
    massloss_all = (
        1 - binstat(subs[xcol][s], subs[ycol][s], "mean", mstarbins).statistic
    )
    massloss_all_lo = binstat(
        subs[xcol][s], subs[ycol][s], error_hi, mstarbins
    ).statistic
    massloss_all_hi = binstat(
        subs[xcol][s], subs[ycol][s], error_lo, mstarbins
    ).statistic
    ax.fill_between(
        x,
        massloss_all - massloss_all_lo,
        massloss_all + massloss_all_hi,
        **kwargs_fill_all,
    )
    ax.plot(x, massloss_all, *args_all, **kwargs_all)
    ax.set(  # ylabel='$1-m_\mathrm{sub}/m_\mathrm{sub}^\mathrm{max}$',
        ylabel="Mass loss", ylim=(0, 1)
    )
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    ax = axes[1]
    ycol = "Mbound"
    msat = binstat2d(
        subs[xcol][s], subs[bincol][s], subs[ycol][s], "mean", (mstarbins, bins)
    ).statistic.T
    msat_lo = binstat2d(
        subs[xcol][s], subs[bincol][s], subs[ycol][s], error_lo, (mstarbins, bins)
    ).statistic.T
    msat_hi = binstat2d(
        subs[xcol][s], subs[bincol][s], subs[ycol][s], error_hi, (mstarbins, bins)
    ).statistic.T
    mcen = binstat(subs[xcol][~s], subs[ycol][~s], "mean", mstarbins).statistic
    tstrip = 1 - msat / mcen
    tstrip_lo = msat_hi / mcen
    tstrip_hi = msat_lo / mcen
    for i, (p, plo, phi, c) in enumerate(zip(tstrip, tstrip_lo, tstrip_hi, colors)):
        ax.fill_between(x, p - plo, p + phi, color=c, **kwargs_fill)
        ax.plot(x, p, "-", color=c, lw=lw, zorder=2)
    # Niemiec+17 definition
    tstrip_n17 = 1 - msat[:3] / msat[-1]
    ic(tstrip_n17)
    for i, (ts, c) in enumerate(zip(tstrip_n17, colors)):
        ax.plot(x, ts, "--", color=c, lw=1.5, zorder=-5)
    msat_all = binstat(subs[xcol][s], subs[ycol][s], "mean", mstarbins).statistic
    msat_all_lo = binstat(subs[xcol][s], subs[ycol][s], error_lo, mstarbins).statistic
    msat_all_hi = binstat(subs[xcol][s], subs[ycol][s], error_hi, mstarbins).statistic
    tstrip_all = 1 - msat_all / mcen
    tstrip_all_lo = msat_all_hi / mcen
    tstrip_all_hi = msat_all_lo / mcen
    ax.fill_between(
        x, tstrip_all - tstrip_all_lo, tstrip_all + tstrip_all_hi, **kwargs_fill_all
    )
    ax.plot(x, tstrip_all, *args_all, **kwargs_all)
    ax.set(
        # ylabel='$m_\mathrm{sat}/m_\mathrm{cen}$',
        ylabel=r"$\tau_\mathrm{strip}$",
        ylim=(0, 1),
    )
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    ax = axes[2]
    bias = tstrip / massloss - 1
    bias_lo = bias * (tstrip_lo**2 + massloss_lo**2) ** 0.5
    bias_hi = bias * (tstrip_hi**2 + massloss_hi**2) ** 0.5
    ic(bias)
    for i, (r, rlo, rhi, c) in enumerate(zip(bias, bias_lo, bias_hi, colors)):
        ax.fill_between(x, r - rlo, r + rhi, color=c, **kwargs_fill)
        ax.plot(x, r, "-", color=c, lw=lw, zorder=2)
    bias_all = tstrip_all / massloss_all - 1
    bias_all_lo = bias_all * (tstrip_all_lo**2 + massloss_all_lo**2) ** 0.5
    bias_all_hi = bias_all * (tstrip_all_hi**2 + massloss_all_hi**2) ** 0.5
    ax.fill_between(
        x, bias_all - bias_all_lo, bias_all + bias_all_hi, **kwargs_fill_all
    )
    ax.plot(x, bias_all, *args_all, **kwargs_all)
    # Niemiec+17
    bias_n17 = tstrip_n17 / massloss[:3] - 1
    ic(bias_n17)
    for i, (b, c) in enumerate(zip(bias_n17, colors)):
        ax.plot(x, b, "--", color=c, lw=1.5, zorder=-5)
    ic(bias_all)
    ax.axhline(0, ls=":", lw=1, color="k")
    ax.set(
        xscale="log",
        xlabel=get_axlabel("Mstar"),
        ylabel="Mass loss bias",
        ylim=(-0.4, 0.8),
    )
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))

    cbar = plt.colorbar(sm, ax=axes, label="$R_{01}/R_\mathrm{200m}$")
    # label=get_label_bincol(bincol))
    cbar.ax.set_yscale("log")
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    output = "massloss/massloss_3pan_dist"
    save_plot(fig, output, sim, tight=False, verbose=True)

    ic(bias_lo, bias_hi)
    # fit bias(R)
    bias_err = (bias_lo + bias_hi) / 2
    w = 1 / bias_err**2
    bias_R = np.sum(w * bias, axis=1) / np.sum(w, axis=1)
    ic(bias_R)
    line = lambda r, a, b: a + b * r
    fit, fitcov = curve_fit(line, np.log10(bin_centers), bias_R)
    ic(fit, fitcov)
    ic(bin_centers)
    ic(line(np.log10(bin_centers), *fit))
    return


def error_lo(x):
    return (x.mean() - np.percentile(x, 16)) / x.size**0.5


def error_hi(x):
    return (np.percentile(x, 84) - x.mean()) / x.size**0.5


if __name__ == "__main__":
    main()
