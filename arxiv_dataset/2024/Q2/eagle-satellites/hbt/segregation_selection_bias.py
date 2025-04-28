from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.gridspec import GridSpec
import multiprocessing as mp
import numpy as np
import os
from time import time
from tqdm import tqdm
import sys

import plottery
from plottery.plotutils import savefig, update_rcParams

update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import binlabel, format_filename, get_label_bincol
from hbtpy.helpers.plot_relations import plot_relation
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)

    reader = HBTReader(sim.path)

    rbins = np.logspace(-2, 0.7, 15)
    rcol = "ComovingMostBoundDistance/R200Mean"
    ycol = "Mbound/history:max_Mbound:Mbound"
    # ycol = 'Mbound/Mstar'

    # test
    # make_plot(args, reader, sim, -1, 11, rcol, ycol, rbins)
    # return

    isnaps = [-1]
    logM200Mean_min = (9, 10, 11, 12, 13)
    if args.ncores > 1:
        pool = mp.Pool(args.ncores)
        _ = [
            pool.apply_async(
                make_plot, args=(args, reader, sim, isnap, logm, rcol, ycol, rbins)
            )
            for logm in logM200Mean_min
            for isnap in isnaps
        ]
        pool.close()
        pool.join()
    else:
        _ = [
            make_plot(args, reader, sim, isnap, logm, rcol, ycol, rbins)
            for logm in logM200Mean_min
            for isnap in isnaps
        ]
    return


def make_plot(args, reader, sim, isnap, logM200Mean_min, rcol, ycol, rbins):
    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logM200Mean_min=logM200Mean_min,
        logMstar_min=None,
        logMmin=None,
        verbose_when_loading=False,
    )
    print(logM200Mean_min, subs.isnap, subs.redshift, subs.satellite_mask.sum())
    # print(np.log10(np.percentile(subs['Mbound'][subs.satellite_mask],
    #                              [0, 1, 25, 50, 75, 99, 100])))
    # sys.exit()

    masscols = ("Mstar", "history:first_infall:Mstar", "history:max_Mbound:Mbound")
    mthresh = (1e9, 1e9, 1e9)
    ls = ("-", "--", "-.")
    mbins = np.logspace(6, 13.3, 51)

    print(
        np.log10(
            np.percentile(
                subs["Mbound"][subs.satellite_mask], [0, 1, 25, 50, 75, 99, 100]
            )
        )
    )
    m = subs["Mbound"]
    ic(m.size)
    sat = subs.satellite_mask
    jmax = subs["history:max_Mbound:Mbound"] > 1e8
    ic(sat.sum(), jmax.sum(), (sat & jmax).sum())
    ic(np.sort(m[sat]))
    ic(np.sort(m[jmax]))
    ic(np.sort(m[sat & jmax]))

    for stat in ("mean", "std"):
        # fig, axes = plt.subplots(
        #     1, 2, figsize=(12,6), constrained_layout=True)
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        # the third is a dummy to acommodate the colorbar label
        gs = GridSpec(3, 4, width_ratios=(8, 0.5, 2, 8), left=0.1, right=0.95)
        ax = fig.add_subplot(gs[:, 0])
        cbar_ax = fig.add_subplot(gs[:, 1])
        haxes = [fig.add_subplot(gs[i, -1]) for i in range(3)]
        # histograms in the right axis
        for i, (hax, mcol) in enumerate(zip(haxes, ("Mbound", "Mstar", "Mdm"))):
            Nzero = [0, 0, 0]
            ncol = f"N{mcol[1:]}"
            for j in range(len(masscols)):
                mask = (subs[masscols[j]] > mthresh[j]) & subs.satellite_mask
                Nzero[j] = str((subs[ncol][mask] <= 1).sum())
                hax.hist(
                    subs[mcol][mask],
                    mbins,
                    color=f"C{j}",
                    lw=0,
                    alpha=0.4,
                    log=True,
                    histtype="stepfilled",
                    zorder=10 - j,
                    label=f"${binlabel[masscols[j]]}$",
                )
            hax.axvline(1e9, ls=":", lw=1, color="k")
            hax.annotate(
                f"${binlabel[mcol]}$\n" + "\n".join(Nzero),
                xy=(0.95, 0.85),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=12,
            )
            hax.set(xscale="log")
            hax.xaxis.set_major_locator(ticker.FixedLocator(np.logspace(6, 12, 4)))
            hax.yaxis.set_major_locator(ticker.FixedLocator([1e1, 1e3, 1e5]))
            # hax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        haxes[2].annotate(
            "Selecting by:",
            xy=(0.03, 0.8),
            xycoords="axes fraction",
            fontsize=11,
            ha="left",
            va="top",
        )
        haxes[2].legend(loc="lower left", fontsize=11, frameon=True)
        haxes[-1].set_xlabel("Mass (M$_\odot$)")
        # haxes[1].set_ylabel('log N')
        for hax in haxes[:-1]:
            hax.xaxis.set_major_formatter(ticker.NullFormatter())
        # cbar_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # cbar_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # main plot
        plot_relation(
            sim,
            subs,
            xcol=rcol,
            ycol=ycol,
            statistic=stat,
            axes=ax,
            bincol="history:first_infall:time",
            bins=np.arange(0, 13, 3),
            binlabel=r"$t_\mathrm{infall}$ (Gya)",
            show_satellites=False,
            yscale="log" if stat == "mean" else "linear",
            cbar_ax=cbar_ax,
            selection=masscols,
            selection_min=mthresh,
            selection_max=(1e20, 1e20, 1e20),
            selection_labels=(
                "$\log m_{\u2605}/$M$_\odot>9$",
                "$\log m_{\u2605}^\mathrm{infall}/$M$_\odot>9$",
                "$\log m_\mathrm{sub}^\mathrm{max}/$M$_\odot>9$",
            ),
            xbins=rbins,
            min_hostmass=None,
            output=False,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # plot y \propto x lines
        if stat == "mean":
            x = np.logspace(-2, 0.6, 10)
            for y in np.logspace(-2, 5, 8):
                ax.plot(x, y * x, "--", color="0.6", lw=1.2)
            if ycol == "Mbound/Mstar":
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set(xlim=xlim, ylim=ylim)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        fig.suptitle(
            f"$\log M_\mathrm{{200m}}/$M$_\odot>{logM200Mean_min}$", fontsize=16
        )
        if isnap < 0:
            isnap = 365 + isnap + 1
        output = format_filename(
            f"segregation__{ycol}__{stat}__M200min_{logM200Mean_min}__"
            f"isnap_{isnap}.pdf"
        )
        output = os.path.join(sim.plot_path, "selection_bias", output)
        savefig(output, fig=fig, tight=False)
    return


main()
