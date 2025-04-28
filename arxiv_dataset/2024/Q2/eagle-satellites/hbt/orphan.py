from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np
import os
from scipy.integrate import trapz
from scipy.stats import gaussian_kde
import seaborn as sns

from plottery.plotutils import savefig, update_rcParams

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import get_axlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

update_rcParams()


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)

    # for z in (0.5, 1, 2):
    #     disrupted(args, sim, reader, z=z)
    # return
    disrupted(args, sim, reader)
    # orphans(args, sim, reader)

    return


def disrupted(args, sim, reader, isnap=-1):
    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logMmin=None,
        logM200Mean_min=13,
        logMstar_min=None,
        exclude_non_FoF=True,
        load_distances=False,
        load_velocities=False,
    )
    cols = [
        "Mstar",
        "Mdm",
        "Nstar",
        "Ndm",
        "M200Mean",
        "Rank",
        "history:first_infall:time",
        "history:max_Mbound:time",
        "history:max_Mbound:Mbound",
        "history:max_Mstar:time",
        "history:max_Mstar:Mstar",
    ]
    subs.sort("Nbound", inplace=True)

    logm0 = np.arange(7, 15, 0.01)
    ## maximal subhalo mass only
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 8), height_ratios=(4, 1), constrained_layout=True
    )
    # n, ndis = disrupted_panel(
    #     axes,
    #     subs,
    #     "history:max_Mbound:Mbound",
    #     np.arange(7, 15, 0.1),
    #     "$m_\mathrm{sub}^\mathrm{max}$ (M$_{\!\odot}\!$)",
    #     xlim=(2e8, 3e14),
    #     ylim=(0.5, 5e6),
    #     kde=False,
    # )
    n, ndis = disrupted_panel(
        axes,
        subs,
        "history:max_Mbound:Mbound",
        logm0,
        "$m_\mathrm{sub}^\mathrm{max}$ (M$_{\!\odot}\!$)",
        xlim=(2e8, 3e14),
        ylim=(0.2, 8e5),
    )
    axes[0].set(ylabel="Number of subhaloes", xticklabels=[])
    axes[1].set(ylabel="Fraction")
    axes[0].legend(fontsize=15)
    axes[1].axhline(0.5, ls="--", color="k", lw=1)
    suff = "max_msub"
    output = os.path.join(sim.plot_path, "orphan", f"orphan__{suff}.pdf")
    savefig(output, fig=fig, tight=False)
    np.savetxt(
        os.path.join(sim.data_path, "disrupted", f"disrupted__{suff}.txt"),
        np.transpose([logm0, n, ndis]),
        fmt="%.3e",
        header="logm0 n_all_mstar n_dis_mstar n_all_msub n_dis_msub",
    )

    ## msub_max and infall stellar mass
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), height_ratios=(4, 1), constrained_layout=True
    )
    n_all_msub, n_dis_msub = disrupted_panel(
        axes[:, 0],
        subs,
        "history:max_Mbound:Mbound",
        logm0,
        "$m_\mathrm{sub}^\mathrm{max}$ (M$_{\!\odot}\!$)",
        xlim=(2e8, 3e14),
        ylim=(0.5, 5e6),
        kde=False,
    )
    n_all_mstar, n_dis_mstar = disrupted_panel(
        axes[:, 1],
        subs,
        "history:first_infall:Mstar",
        logm0,
        "$m_{\u2605}^\mathrm{infall}$ (M$_{\!\odot}\!$)",
        xlim=(8e6, 3e12),
        kde=False,
    )
    axes[0, 0].set(ylabel="Number of subhaloes")
    axes[1, 0].set(ylabel="Fraction")
    for ax in axes[0]:
        ax.legend(fontsize=20)
    for i, ax in enumerate(axes[:, 1]):
        ax.set_ylim(axes[i, 0].get_ylim())
        ax.set_yticklabels([])
    for ax in axes[1]:
        ax.axhline(0.5, ls="--", color="k", lw=1, label="Cumulative")
    for ax in axes[:, 1]:
        ax.xaxis.set_major_locator(ticker.FixedLocator(10 ** np.arange(8, 15, 1)))
    suff = "max_msub__mstar_infall"
    output = os.path.join(sim.plot_path, "orphan", f"orphan__{suff}.pdf")
    savefig(output, fig=fig, tight=False)
    # n = np.transpose([logm0, n_all_msub, n_dis_msub, n_all_mstar, n_dis_mstar])
    # np.savetxt(
    #     os.path.join(sim.data_path, "disrupted", f"disrupted__{suff}.txt"),
    #     n,
    #     fmt="%.3e",
    #     header="logm0 n_all_mstar n_dis_mstar n_all_msub n_dis_msub",
    # )
    return


def disrupted_panel(
    axes,
    subs,
    col,
    logm0,
    xlabel,
    xlim=None,
    ylim=None,
    Ncol="Nbound",
    Nlabel="particles",
    Nmin=20,
    kde=True,
    bw_method=0.1,
):
    logm = np.log10(subs[col])
    all = (logm >= logm0[0]) & (logm <= logm0[-1])
    disrupted = all & (subs[Ncol] < Nmin)
    ic((logm[all] > 14).sum())
    kde_all = gaussian_kde(logm[all], bw_method)
    area_all = trapz(kde_all(logm0), logm0)
    n_all = all.sum() * kde_all(logm0) / area_all
    kde_dis = gaussian_kde(logm[disrupted])
    area_dis = trapz(kde_dis(logm0), logm0)
    n_dis = disrupted.sum() * kde_dis(logm0) / area_dis
    # this makes it such that there is a peak at y=1 for m>1e14
    norm = n_all[logm0 > 14].max()
    n_all /= norm
    n_dis /= norm
    ax = axes[0]
    if kde:
        ax.plot(10**logm0, n_all, "C0-", lw=3, label="All satellites")
        ax.plot(
            10**logm0, n_dis, "C1-", lw=3, label=f"$N_\mathrm{{{Nlabel}}}<{Nmin}$"
        )
    else:
        ax.hist(10 ** logm[all], 10**logm0, color="C0", histtype="step")
        ax.hist(10 ** logm[disrupted], 10**logm0, color="C1", histtype="step")
    # n_all = np.histogram(logm[all], logm0)[0]
    # n_dis = np.histogram(logm[disrupted], logm0)[0]
    # logm0 = (logm0[1:] + logm0[:-1]) / 2
    ratio = n_dis / n_all
    # ratio(>m)
    ratio_cum = (np.cumsum(n_dis[::-1]) / np.cumsum(n_all[::-1]))[::-1]
    ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim, xticklabels=[])
    ax = axes[1]
    ax.plot(10**logm0, ratio, "k-", lw=3)
    ax.plot(10**logm0, ratio_cum, "-", color="0.5", lw=3, label="f(>m)")
    ax.set(xlabel=xlabel, xscale="log", xlim=xlim)
    ax.set(ylim=(0, 1))
    return n_all, n_dis


def orphans(args, sim, reader, logMstar_min=6, isnap=-1):
    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logMmin=None,
        logM200Mean_min=None,
        logMstar_min=None,
        exclude_non_FoF=False,
        load_distances=False,
        load_velocities=False,
    )
    cols = [
        "Mstar",
        "Mdm",
        "Nstar",
        "Ndm",
        "M200Mean",
        "Rank",
        "history:first_infall:time",
        "history:max_Mbound:time",
        "history:max_Mbound:Mbound",
    ]
    subs.sort("NboundType1", inplace=True)
    Ndm_max = 1
    mask = subs["Ndm"] <= Ndm_max
    print(f"{mask.sum()} subhaloes with Ndm <= {Ndm_max}:")
    print(subs[cols][mask])

    Nmin = (1, 20, 100)
    mstarbins = np.logspace(logMstar_min, 11, 31)
    # the 0.5 makes it clear where each N falls
    nbins = np.append(np.array([-1, 0, 1, 2, 5]), np.arange(10, 101, 10)) + 0.5
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    ax = axes[0]
    bottom = None
    m200min = np.logspace(9, 13, 3)
    for i, m in enumerate(m200min):
        j = subs["M200Mean"] > m
        h = np.histogram(subs["Ndm"][j], nbins)[0]
        label = rf"$M_\mathrm{{200m}}>10^{{{np.log10(m):.0f}}}\,\mathrm{{M}}_\odot$"
        ax.bar(np.arange(h.size), h, label=label, width=0.9 - 0.2 * i)
    ax.annotate(
        "$N_\mathrm{DM}$",
        xy=(0.2, 0.9),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=18,
    )
    ax.legend(fontsize=14, loc="upper right")
    xlabels = ["N = 0", "N = 1", "N = 2"] + [
        f"{nbins[i-1]-0.5:.0f} < N \leq {nbins[i]-0.5:.0f}"
        for i in range(4, nbins.size)
    ]
    xlabels = [f"${label}$" for label in xlabels]
    ax.set(
        # ylabel="$N(N_\mathrm{DM})$",
        xticks=np.arange(h.size),
        yscale="log",
        # ylim=(0.8, 1500),
    )
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.set_xticklabels(xlabels, rotation=90)
    print(ax.get_xticks())

    ax = axes[1]
    ax.hist(subs["Mstar"], mstarbins, histtype="step", lw=2, label="All subhaloes")
    for i, Nmin_i in enumerate(Nmin):
        mask = subs["Ndm"] <= Nmin_i
        kwargs = dict(histtype="step", color=f"C{i+1}", zorder=10 - i)
        ax.hist(
            subs["Mstar"][mask & subs.satellite_mask],
            mstarbins,
            lw=3,
            label=f"$N_\mathrm{{DM}}\leq{Nmin_i}$",
            **kwargs,
        )
        ax.hist(subs["Mstar"][mask & subs.central_mask], mstarbins, lw=1.5, **kwargs)
        ic(Nmin_i)
        for mstar_min in (1e7, 1e8, 1e9):
            j = (subs["Mstar"] >= mstar_min) & mask
            ic(
                np.log10(mstar_min),
                j.size,
                j.sum(),
                subs[["Rank", "Nbound", "Ndm"]][j],
                subs[["Mdm", "Mstar"]][j] / 1e8,
                subs["Mdm/Mstar"][j],
            )
    ax.set(xscale="log", yscale="log", xlabel="Mstar (M$_\odot$)")  # , ylabel="N")
    ax.legend(fontsize=14, loc="upper right")
    ax.xaxis.set_major_locator(
        ticker.FixedLocator(
            np.logspace(int(logMstar_min), 11, 11 - int(logMstar_min) + 1)
        )
    )
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.logspace(0, 6, 7)))
    #
    mkeys = ("history:max_Mstar:Mstar", "LastMaxMass")
    xlabels = ("$m_{\u2605}^\mathrm{max}$ (M$_\odot$)", "LastMaxMass (M$_\odot$)")
    for ax, mkey, xlabel in zip(axes[2:], mkeys, xlabels):
        logmmin = 8
        mbins = np.logspace(logmmin, 14, 31)
        ax.hist(subs[mkey], mbins, histtype="step", lw=2, label="All subhaloes")
        for i, Nmin_i in enumerate(Nmin):
            mask = subs["Ndm"] <= Nmin_i
            kwargs = dict(histtype="step", color=f"C{i+1}", zorder=10 - i)
            ax.hist(
                subs[mkey][mask & subs.satellite_mask],
                mbins,
                lw=3,
                label=f"$N_\mathrm{{DM}}\leq{Nmin_i}$",
                **kwargs,
            )
            ax.hist(
                subs[mkey][mask & subs.central_mask],
                mbins,
                lw=1.5,
                **kwargs,
            )
        ax.set(xscale="log", yscale="log", xlabel=xlabel)
        # ax.legend(fontsize=14)
        ax.xaxis.set_major_locator(
            ticker.FixedLocator(np.logspace(int(logmmin), 14, 14 - int(logmmin) + 1))
        )
    output = os.path.join(sim.plot_path, "orphan.png")
    savefig(output, fig=fig, tight=False)


main()
