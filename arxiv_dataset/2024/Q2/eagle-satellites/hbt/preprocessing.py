"""Multi-panel figure highlighting the role of preprocessing"""
import cmasher as cmr
from icecream import ic
from matplotlib import cm, colors as mplcolors, patheffects as pe, pyplot as plt, ticker
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binstat, binned_statistic_2d as binstat2d
import sys
from time import time
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams

update_rcParams()

from hbtpy import hbt_tools
from hbtpy.hbt_tools import format_colname, load_subhalos, save_plot
from hbtpy.subhalo import Subhalos
from hbtpy.helpers.plot_auxiliaries import (
    get_axlabel,
    get_bins,
    get_label_bincol,
    logcenters,
    plot_line,
)
from hbtpy.helpers.plot_relations import plot_relation_lines, relation_lines


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
        logMstar_min=None,
        logMmin=None,
        logM200Mean_min=None,
    )
    s = subs.satellite_mask

    # default - SHSMR
    control = subs["history:first_infall:Depth"] == 0
    mask = subs["history:sat:time"] < 12
    samples = [
        [subs["M200Mean"] > 1e14],
        [(subs["M200Mean"] > 5e13) & (subs["M200Mean"] < 1e14)],
        [subs["M200Mean"] < 5e13],
    ]
    output_suffix = ["massivehost", "midmasshost", "smallhost"]
    if args.ncores == 1:
        _ = [
            make_plot_large(
                args,
                subs,
                control,
                statistic=stat,
                mask=mask,
                samples=sample,
                output_suffix=suff,
            )
            for stat in ("mean", "std")
            for sample, suff in zip(samples, output_suffix)
        ]
    else:
        pool = Pool(args.ncores)
        for stat in ("mean", "std"):
            [
                pool.apply_async(
                    make_plot_large,
                    args=(args, subs, control),
                    kwds={
                        "statistic": stat,
                        "mask": mask,
                        "samples": sample,
                        "output_suffix": suff,
                    },
                )
                for sample, suff in zip(samples, output_suffix)
            ]
        pool.close()
        pool.join()
    return

    h = "history:first_infall"
    xcols = ["Mstar", f"{h}:Mstar"]
    ycols = ["Mbound/Mstar", f"{h}:Mbound/{h}:Mstar"]
    for xcol, ycol in zip(xcols, ycols):
        logbins = "time" not in xcol
        # this may not be necessary in other circumstances
        mask = subs["history:sat:time"] < 12
        # central infallers
        jcen = subs["history:first_infall:Depth"] == 0
        ic(jcen.size, jcen.sum())
        bincol = "history:first_infall:time"
        panel_bins = np.array([0, 4, 8, 12])
        sbincol = "history:sat:time-history:first_infall:time"
        # sbins = np.arange(-2, 12.1, 2)
        sbins = [-1, 0.1, 1, 2, 4, 12]
        for statistic, n_xbins in zip(("mean", "std"), (8, 6)):
            if ("Mstar" in xcol) and ("/" not in xcol):
                xbins = np.logspace(8.9, 11.3, 9)
            else:
                xbins = get_bins(xcol, logbins, n=n_xbins)
            if logbins:
                xcenters = logcenters(xbins)
            else:
                xcenters = (xbins[:-1] + xbins[1:]) / 2
            make_plot(
                args,
                subs,
                xcol,
                ycol,
                bincol,
                sbincol,
                xbins,
                xcenters,
                panel_bins,
                sbins,
                jcen,
                statistic=statistic,
                mask=mask,
            )
            # break

    return


def make_plot_large(
    args,
    subs,
    control,
    xcol="Mstar",
    ycol="Mbound/Mstar",
    bincol="history:first_infall:time",
    sbincol="history:sat:time-history:first_infall:time",
    events=("first_infall", "sat"),
    xbins=np.logspace(8.9, 11.3, 9),
    xcenters=None,
    bins=np.array([0, 4, 8, 12]),
    sbins=[-1, 0.1, 1, 2, 4, 12],
    control_label="central\ninfallers",
    mask=None,
    samples=None,
    statistic="mean",
    cmap="cmr.voltage",
    cmap_range=(0.1, 0.8),
    output_suffix="",
):
    """Default kwargs are appropriate for SHSMR"""
    # brute-force for now
    if statistic == "std":
        xbins = np.logspace(8.9, 11.3, 6)
    ncols = len(bins) - 1
    nrows = len(events) + 1
    fig = plt.figure(figsize=(4.5 * ncols, 5 * nrows), constrained_layout=True)
    # this is fixed to 3 rows...
    gs = fig.add_gridspec(nrows + 1, ncols, figure=fig, height_ratios=[1, 12, 12, 12])
    cax = fig.add_subplot(gs[0, :])
    axes = np.array(
        [[fig.add_subplot(gs[i + 1, j]) for j in range(nrows)] for i in range(ncols)]
    )
    xcols = [xcol] + [f"history:{h}:Mstar" for h in events]
    ycols = ["Mbound/Mstar"]
    ycol = ycol.split("/")
    ycols = ycols + [f"/".join([f"history:{h}:{y}" for y in ycol]) for h in events]
    ic(xcols, ycols)
    if xcenters is None:
        if "time" in xcol:
            xcenters = (xbins[1:] + xbins[:-1]) / 2
        else:
            xcenters = logcenters(xbins)

    # colormap
    sbins_rng = np.arange(len(sbins))
    sbincenters = (sbins_rng[:-1] + sbins_rng[1:]) / 2
    nbins = len(sbins) + 1
    cmap = cmr.get_sub_cmap(cmap, *cmap_range)
    cmap = plt.get_cmap(cmap, nbins)
    ic(cmap)
    boundary_norm = mplcolors.BoundaryNorm(sbins_rng, cmap.N)
    ic(boundary_norm)
    sm = cm.ScalarMappable(cmap=cmap, norm=boundary_norm)
    sm.set_array([])
    colors = sm.to_rgba(sbincenters)
    ic(colors)

    ylims = [(3, 200), (10, 200), (10, 200)]
    kwds_bintext = {
        "xy": (0.5, 0.95),
        "xycoords": "axes fraction",
        "va": "top",
        "ha": "center",
        "fontsize": 17,
    }
    # logfile = os.path.join(
    #     subs.sim.data_path,
    #     "preprocessing",
    #     f"preprocessing__{format_colname(ycol)}__{output_suffix}.txt",
    # )
    for row, xcol, ycol, ylim in zip(axes, xcols, ycols, ylims):
        x = subs[xcol]
        y = subs[ycol]
        b = subs[bincol]
        s = subs[sbincol]
        if mask is None:
            mask = np.ones(x.size, dtype=bool)
        row[0].set(ylabel=get_axlabel(ycol, statistic))
        relations = plot_row(
            args,
            row,
            subs,
            x,
            y,
            b,
            s,
            mask,
            xbins,
            xcenters,
            bins,
            sbins,
            control,
            statistic,
            samples,
            colors,
            get_axlabel(xcol),
            get_axlabel(ycol),
            ylim=ylim,
        )
        # assumes we are showing central galaxies
        y_ceninfall, y_satinfall, y_centrals = relations
        ic(
            output_suffix,
            xcol,
            xbins,
            y_centrals,
            y_ceninfall,
            y_ceninfall / y_centrals,
            y_satinfall,
            output_suffix,
            xcol,
            y_satinfall / y_centrals[:, None],
            y_satinfall / y_ceninfall[:, None],
            np.nanmedian(y_ceninfall / y_centrals, axis=1),
            np.nanmedian(y_satinfall / y_centrals[:, None], axis=2),
            np.nanmedian(y_satinfall / y_ceninfall[:, None], axis=2),
        )
        for i, ax in enumerate(row):
            ax.annotate(
                f"${bins[i]} < t_\mathrm{{infall}}/\mathrm{{Gya}} < {bins[i+1]}$",
                **kwds_bintext,
            )
    # again this is hard-coded
    if "massivehost" in output_suffix:
        note = "$M_\mathrm{200m} > 10^{14}\,$M$_\odot$"
    elif "midmasshost" in output_suffix:
        note = "$5\\times10^{13} < M_\mathrm{200m}/$M$_\odot < 10^{14}$"
    elif "smallhost" in output_suffix:
        note = "$M_\mathrm{200m} < 5\\times10^{13}\,$M$_\odot$"
    xy = (0.05, 0.2) if statistic == "mean" else (0.05, 0.7)
    axes[-1, 0].annotate(
        note,
        xy=xy,
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=18,
    )
    # colorbar
    cbar = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        location="top",
        label=get_label_bincol(sbincol),
        aspect=30,
        fraction=0.8,
    )
    cbar.ax.set_xticks(
        [sbins_rng[0]] + [sum(sbins_rng[:2]) / 2] + list(sbins_rng[1:]),
        ["", "Direct\ninfallers"] + sbins[1:],
    )
    cbar.ax.tick_params(length=0, labelsize=16)
    # output = set_output_name(
    #     xcol, ycol, bincol, sbincol, statistic, prefix="preprocessing_large"
    # )
    output = f"preprocessing/preprocessing_large_{statistic}"
    if output_suffix:
        output = f"{output}_{output_suffix}"
    # output = f"preprocessing/preprocessing_large_{statistic}_smallhost"
    save_plot(fig, output, subs.sim, tight=False, verbose=True)
    # sys.exit()
    return


def make_plot(
    args,
    subs,
    xcol,
    ycol,
    bincol,
    sbincol,
    xbins,
    xcenters,
    panel_bins,
    sbins,
    jcen,
    statistic="mean",
    mask=None,
    cmap="cmr.voltage",
    cmap_range=(0.1, 0.8),
    show_massive_only=True,
):
    """jcen is the mask of central infallers"""
    xdata = subs[xcol]
    ydata = subs[ycol]
    bindata = subs[bincol]
    sbindata = subs[sbincol]
    # in massive/low-mass clusters
    massive_host = subs["M200Mean"] > 1e14
    lowmass_host = subs["M200Mean"] < 5e13
    if mask is None:
        mask = np.ones(xdata.size, dtype=bool)

    sbins_rng = np.arange(len(sbins))
    sbincenters = (sbins_rng[:-1] + sbins_rng[1:]) / 2
    nbins = len(sbins) + 1
    cmap = cmr.get_sub_cmap(cmap, *cmap_range)
    cmap = plt.get_cmap(cmap, nbins)
    ic(cmap)
    boundary_norm = mplcolors.BoundaryNorm(sbins_rng, cmap.N)
    ic(boundary_norm)
    sm = cm.ScalarMappable(cmap=cmap, norm=boundary_norm)
    sm.set_array([])
    colors = sm.to_rgba(sbincenters)
    ic(colors)

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 5.5), constrained_layout=True, sharey=True
    )
    plot_row(
        args,
        axes,
        subs,
        xdata,
        ydata,
        bindata,
        sbindata,
        mask,
        xbins,
        xcenters,
        panel_bins,
        sbins,
        subs["history:first_infall:Depth"] == 0,
        statistic,
        [subs["M200Mean"] > 1e14],
        colors,
        get_axlabel(xcol),
        get_axlabel(ycol, statistic),
    )
    axes[0].set(ylabel=get_axlabel(ycol, statistic))
    # colorbar
    cbar = plt.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        location="top",
        label=get_label_bincol(sbincol),
        aspect=30,
        fraction=0.8,
    )
    cbar.ax.set_xticks(
        [sbins_rng[0]] + [sum(sbins_rng[:2]) / 2] + list(sbins_rng[1:]),
        ["", "central\ninfallers"] + sbins[1:],
    )
    cbar.ax.tick_params(length=0, labelsize=16)
    text_kwds = {
        "xy": (0.5, 0.95),
        "xycoords": "axes fraction",
        "va": "top",
        "ha": "center",
        "fontsize": 17,
    }
    # generalize later
    axes[0].annotate("$0 < t_\mathrm{infall}/\mathrm{Gya} < 4$", **text_kwds)
    axes[1].annotate("$4 < t_\mathrm{infall}/\mathrm{Gya} < 8$", **text_kwds)
    axes[2].annotate("$8 < t_\mathrm{infall}/\mathrm{Gya} < 12$", **text_kwds)
    output = set_output_name(xcol, ycol, bincol, sbincol, statistic)
    save_plot(fig, output, subs.sim, tight=False, verbose=True)
    return


def plot_row(
    args,
    axes,
    subs,
    xdata,
    ydata,
    bindata,
    sbindata,
    mask,
    xbins,
    xcenters,
    bins,
    sbins,
    control,
    statistic,
    samples,
    colors,
    xlabel,
    ylabel,
    show_centrals=True,
    ylim=None,
):
    rel_centrals = []
    rel_ceninfall = []
    rel_satinfall = []
    for i, ax in enumerate(axes):
        binmask = mask & (bindata > bins[i]) & (bindata <= bins[i + 1])
        binmean = np.nanmedian(bindata[binmask])
        if show_centrals:
            # this assumes that the bincol is a lookback time!!
            jsnap = np.argmin(np.abs(binmean - subs.sim.t_lookback.value))
            ic(jsnap, subs.sim.t_lookback[jsnap], subs.sim.redshifts[jsnap])
            # maybe we should actually show the past SHSMR of galaxies that
            # are still centrals today -- use subs.merge on TrackId
            sim_j, subs_j = load_subhalos(args, isnap=jsnap)
            cen_past = Subhalos(
                subs_j,
                subs.sim,
                jsnap,
                load_velocities=False,
                load_distances=False,
                verbose_when_loading=False,
                logMstar_min=None,
                logMmin=None,
                logM200Mean_min=None,
            )
            cen_past.merge(
                subs[["TrackId", "Rank", "Mstar", "Mbound/Mstar"]][subs.central_mask],
                left_cols=["TrackId", "Rank", "Mstar", "Mbound/Mstar"],
                on="TrackId",
                how="right",
                suffixes=("", "_0"),
            )
            ic(
                cen_past[
                    "TrackId",
                    "Rank",
                    "Mstar",
                    "Mbound/Mstar",
                    "Rank_0",
                    "Mstar_0",
                    "Mbound/Mstar_0",
                ]
            )
            rel_centrals.append(
                relation_lines(
                    cen_past["Mstar"],
                    cen_past["Mbound/Mstar"],
                    xbins,
                    statistic,
                    return_err=False,
                )
            )
            ic(xcenters, rel_centrals[-1])
            plot_line(ax, xcenters, rel_centrals[-1], ls="o--", color="k", lw=3)
            ax.plot([], [], "o--", color="k", lw=3, label=f"Centrals {binmean:.1f} Gya")
            # this is quite forced
            loc = (
                "center left"
                if (statistic == "std" and "sat" in xlabel)
                else "lower left"
            )
            ax.legend(loc=loc, fontsize=14)
        ic(binmask.sum())
        cenmask = [binmask & control]
        satmask = [binmask & (~control)]
        if len(samples) > 1:
            for sample in samples[1:]:
                cenmask.append(cenmask[0] & sample)
                satmask.append(satmask[0] & sample)
        cenmask[0] = cenmask[0] & samples[0]
        satmask[0] = satmask[0] & samples[0]
        for cen, sat, lw, alpha in zip(cenmask, satmask, (3, 2), (0.25, 0)):
            # note the [0] at the end, which means we're only appending the mean relations and not the uncertainties. This will fail if we use alpha=0
            rel_ceninfall.append(
                plot_relation_lines(
                    xdata,
                    ydata,
                    xbins,
                    xcenters,
                    statistic,
                    cen,
                    None,
                    None,
                    ax,
                    colors[:1],
                    ls="-",
                    lw=lw,
                    alpha_uncertainties=alpha,
                )[0]
            )
            # satellite infallers -- remember that the first sbin is only
            # to capture central infallers in the colormap
            rel_satinfall.append(
                plot_relation_lines(
                    xdata,
                    ydata,
                    xbins,
                    xcenters,
                    statistic,
                    sat,
                    sbindata,
                    sbins[1:],
                    ax,
                    colors[1:],
                    ls="-",
                    lw=lw,
                )[0]
            )
        # formatting
        ax.set(xscale="log", xlabel=xlabel)
        if statistic == "mean":
            ax.set(yscale="log", ylim=ylim)
        else:
            ax.set(ylim=(0, 0.6))
        if statistic == "mean":
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        for ax in axes[1:]:
            ax.set(ylim=axes[0].get_ylim(), yticklabels=[])
    rel_ceninfall = np.array(rel_ceninfall)
    rel_satinfall = np.array(rel_satinfall)
    if show_centrals:
        return rel_ceninfall, rel_satinfall, np.array(rel_centrals)
    return rel_ceninfall, rel_satinfall


def set_output_name(xcol, ycol, bincol, sbincol, statistic, prefix="preprocessing"):
    xcol = format_colname(xcol)
    ycol = format_colname(ycol)
    statistic = statistic.replace("/", "-over-")
    output = f"{prefix}__{xcol}__{ycol}"
    if bincol is not None:
        bincol = format_colname(bincol)
        output = f"{output}__{bincol}"
    if sbincol is not None:
        sbincol = format_colname(sbincol)
        output = f"{output}__{sbincol}"
    output = f"{output}_{statistic}"
    output = os.path.join("preprocessing", f"{output}")
    return output


if __name__ == "__main__":
    main()
