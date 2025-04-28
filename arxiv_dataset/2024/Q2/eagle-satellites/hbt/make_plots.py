from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from icecream import ic
import matplotlib as mpl
from matplotlib import colors as mplcolors, pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic as binstat
import sys
from time import time
from tqdm import tqdm

import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

from plottery.plotutils import savefig, update_rcParams

update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import load_subhalos, save_plot, timer
from hbtpy.hbtplots import RelationPlotter
from hbtpy.helpers import plot_massfuncs as pmf, plot_relations as pr
from hbtpy.helpers.plot_auxiliaries import definitions, logcenters, massbins
from hbtpy.helpers.plot_definitions import xbins, binlabel

# from hbtpy.hbtplots.core import ColumnLabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos


def main():
    # these arguments allow us to choose easily which set of
    # relations to run
    args = (
        (
            "--relations",
            {
                "choices": ("distance", "hsmr", "hsmr-history", "time"),
                "nargs": "+",
                "default": ["hsmr"],
            },
        ),
        (
            "--stats",
            {
                "choices": ("count", "mean", "std", "std/mean"),
                "nargs": "+",
                "default": ["mean", "std"],
            },
        ),
    )
    args = hbt_tools.parse_args(args=args)
    isnap = -1
    sim, subs = load_subhalos(args, isnap=isnap)
    wrap_plot(args, sim, subs, isnap)
    return


def wrap_plot(
    args,
    sim,
    subs,
    isnap,
    hostmass="logM200Mean",
    logM200Mean_min=None,
    logMmin=None,
    debug=True,
    logMstar_min=9,
    do_plot_relations=True,
    do_plot_massfunctions=False,
):
    subs = Subhalos(
        subs,
        sim,
        isnap,
        exclude_non_FoF=True,
        logMmin=logMmin,
        logM200Mean_min=logMmin,
        logMstar_min=logMstar_min,
        verbose_when_loading=False,
    )
    print(np.sort(subs.colnames))
    print("{0} objects".format(subs[subs.colnames[0]].size))
    print()
    """
    plotter = RelationPlotter(sim, subs)
    for col in ('Mbound', 'Mstar', 'ComovingMostBoundDistance0',
                'Mbound/M200Mean', 'Mbound/Mstar',
                'Mstar/ComovingMostBoundDistance0'):
        ic(col, plotter.label(col))
    for event in ('first_infall', 'last_infall', 'cent', 'sat'):
        for col in ('Mbound', 'Mstar'):
            col = f'history:{event}:{col}'
            ic(col, event)
            ic(plotter.label(col))
    ic(plotter.label('history:last_infall:Mbound/history:first_infall:Mstar'))
    col = 'Mbound'
    for st in ('mean', 'std', 'std/mean'):
        ic(col, plotter.axlabel(col, statistic=st))
    return
    """

    # plot phase-space
    # plot_rv(sim, subs)
    # return
    # test_velocities(subs)

    ### HSMR relations ###
    if do_plot_relations:
        pr.run(args, sim, subs, logM200Mean_min, args.relations, ncores=args.ncores)

    ### mass functions ###
    if do_plot_massfunctions:
        for norm in (True, False):
            plot_massfunction2(
                sim,
                subs,
                "M200Mean",
                xbins["M200Mean"],
                r"$M_\mathrm{{200m}}$ ($h^{{-1}}{0}$)".format(Msun),
                norm=norm,
            )
            bincol = "ComovingMostBoundDistance"
            for log in (True, False):
                pref = "log" * log
                rbins = xbins[f"{pref}{bincol}"]
                plot_massfunction2(
                    sim,
                    subs,
                    bincol,
                    bins=rbins,
                    binlabel=binlabel[bincol],
                    norm=norm,
                    bin_in_log10=log,
                )

    return

    plot_occupation(sim, subs)
    mass_sigma_host(sim, subs)
    return

    if norm:
        print("xyz =", xyz.shape, subhalos["R200MeanComoving"].shape)
        xyz = xyz / np.array(subhalos["R200MeanComoving"])
        vhostkey = "Physical{0}HostVelocityDispersion".format(vref)
        v = v / np.array(subhalos[vhostkey])
    # for ax, projection in zip(projections):

    return


################################################
################################################


def dark_fraction(sim, subs, mhost_norm=True, plot_path=None):
    if mhost_norm:
        mbins = np.logspace(-4, -0.3, 21)
    else:
        mbins = massbins(sim, mtype="total")
    mx = (mbins[:-1] + mbins[1:]) / 2
    m = definitions(subs, as_dict=True)
    dark = np.array(m["dark"], dtype=np.uint8)
    x = m["mtot"] / m["mhost"] if mhost_norm else m["mtot"]
    Ndark = np.histogram(x, bins=mbins, weights=dark)[0]
    Ngal = np.histogram(x, bins=mbins, weights=1 - dark)[0]
    ndark = Ndark / np.histogram(x, bins=mbins)[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(np.log10(mx), ndark)
    if mhost_norm:
        xlabel = r"$\log\mu\equiv\log m_\mathrm{sub}/M_\mathrm{host}$"
    else:
        label = r"log $m_\mathrm{sub}/$M$_\odot$"
    ax.set(xlabel=xlabel, ylabel="Fraction of dark satellites")
    output = "dark_fraction"
    if mhost_norm:
        output += "_mhost"
    if plot_path:
        output = os.path.join(plot_path, output)
    save_plot(fig, output, sim)
    return


def mass_sigma_host(sim, subs):
    # mass bins (defined this way so that central values are
    # equally spaced in log space)
    mbins = np.linspace(10, 15, 20)
    m = 10 ** ((mbins[:-1] + mbins[1:]) / 2)
    mbins = 10**mbins
    fig, axes = plt.subplots(figsize=(15, 5), ncols=2)
    # fig = plt.figure(figsize=(12,5))
    # axes = [plt.subplot2grid((1,15), (0,0), colspan=7),
    # plt.subplot2grid((1,15), (0,7), colspan=8)]
    h = (subs.catalog["Rank"] == 0) & (subs.catalog["Nsat"] > 20)
    print("Using {0} halos".format(h.sum()))
    ax = axes[0]
    sigma3d = subs.sigma()[h]
    mtot = subs.mass("total")[h]
    nsat = subs.catalog["Nsat"][h]
    sigma_mean = binstat(mtot, sigma3d, "mean", mbins)[0]
    sigma_median = binstat(mtot, sigma3d, "median", mbins)[0]
    c = ax.scatter(
        mtot, sigma3d, marker=",", s=1, c=np.log10(nsat), cmap="viridis", label="_none_"
    )
    plt.colorbar(c, ax=ax, label=r"log $N_\mathrm{sat}$")
    ax.plot(m, sigma_mean, "C0-", label="Mean")
    ax.plot(m, sigma_median, "C1--", label="Median")
    ax.set_ylabel("${0}$ (km/s)".format(subs.slabel()))
    ax.legend()
    ax = axes[1]
    c = ax.scatter(
        mtot, subs.sigma(0)[h], marker=",", s=2, c=np.log10(nsat), cmap="viridis"
    )
    plt.colorbar(c, ax=ax, label=r"log $N_\mathrm{sat}$")
    ax.plot(m, sigma_mean / 3**0.5, "C0-")
    ax.plot(m, sigma_median / 3**0.5, "C1--")
    # by hand for now, while isnap=-1 in eagle (shouldn't be too
    # different in other sims)
    ax.plot(m, munari13(m, sim.cosmology, z=0), "C2-", lw=3, label="Munari+13")
    ax.set_ylabel("${0}$ (km/s)".format(subs.slabel(0)))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("${0}$ (M$_\odot$)".format(sim.masslabel(mtype="total")))
    output = "mass_sigma"
    out = os.path.join(sim.plot_path, "{0}.pdf".format(output))
    savefig(out, fig=fig)
    return


def plot_rv(sim, subs, hostmass="M200Mean", weights=None):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = definitions(
        subs, hostmass=hostmass
    )
    # use subs attributes for distance and velocity
    testID = subs.catalog["HostHaloId"].iloc[10000]
    j = subs.catalog["HostHaloId"] == testID
    print("testID =", testID, j.sum())
    print("calculating r and v")
    r = (subs.catalog["ComovingMostBoundDistance"] / subs.catalog["R200MeanComoving"])[
        sat
    ]
    v = (
        subs.catalog["PhysicalMostBoundPeculiarVelocity"]
        / subs.catalog["PhysicalMostBoundHostVelocityDispersion"]
    )[sat]
    rbins = np.arange(0, 3.01, 0.05)
    vbins = np.arange(-5, 5.01, 0.1)
    print("Calculating histograms")
    hist2d, xe, ye = np.histogram2d(r, v, (rbins, vbins))
    # hist2d[hist2d == 0] = np.nan
    # hist2d.T
    print("hist2d =", hist2d.shape)
    # return
    extent = (xe[0], xe[-1], ye[0], ye[-1])
    print("Making plot")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        hist2d.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        norm=mpl.colors.LogNorm(),
    )
    plt.colorbar(im, ax=ax, label=r"$N_\mathrm{sat}$")
    ax.set_xlabel(r"$")
    ax.set_ylabel(r"$v_\mathrm{pec}/\sigma_{v,\mathrm{host}}$")
    output = os.path.join(sim.plot_path, "phase-space", "rv_3d.pdf")
    print("Saving plot")
    savefig(output, fig=fig)
    print("Finished!")
    return


def plot_occupation(sim, subs, hostmass="M200Mean"):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = definitions(subs)
    # binning and plot
    mbins = np.logspace(8, 15, 41)
    m = logcenters(mbins)
    msbins = np.logspace(5.5, 12.5, 26)
    ms = logcenters(msbins)
    mlim = np.log10(m[:: m.size - 1])
    mslim = np.log10(ms[:: ms.size - 1])
    nh = np.histogram(mhost[cen], mbins)[0]
    nc = np.histogram2d(mstar[cen & ~dark], mhost[cen & ~dark], (msbins, mbins))[0]
    nsat = np.histogram2d(mstar[sat & ~dark], mhost[sat & ~dark], (msbins, mbins))[0]
    nsub = np.histogram2d(mstar[sat & ~dark], mtot[sat & ~dark], (msbins, mbins))[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    # ax.plot(m, nh_mtot, 'k-', label='Halos')
    # ax.plot(m, np.sum(nc, axis=0), 'C0:', label='Total Centrals')
    # ax.plot(m, np.sum(nsat, axis=0), 'C1:', label='Total Satellites')
    # ax.hist(mhost[cen & ~dark], mbins, histtype='step', alpha=0.5, color='C3')
    nc = nc / nh
    nsat = nsat / nh
    ax.plot(m, np.sum(nc, axis=0), "C0-", label="Centrals")
    ax.plot(m, np.sum(nsat, axis=0), "C1-", label="Satellites")
    # just a narrow bin
    mmin = 10
    mmax = 11
    logmsbins = np.log10(msbins)
    jmin = np.argmin((logmsbins - mmin) ** 2)
    jmax = np.argmin((logmsbins - mmax) ** 2)
    j = np.arange(jmin, jmax + 1, 1, dtype=int)
    ax.plot(m, np.sum(nc[j], axis=0), "C0--")
    ax.plot(m, np.sum(nsat[j], axis=0), "C1--")
    ax.plot(
        [],
        [],
        "k--",
        label=r"$\log {2}$ in [{0:.2f},{1:.2f})".format(
            logmsbins[j][0], logmsbins[j][-1], sim.masslabel(mtype="stars")
        ),
    )
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    mhlabel = r"M_\mathrm{{{0}}}".format(hostmass[1:5].lower())
    ax.set_xlabel(r"${0}$".format(mhlabel))
    ax.set_ylabel(r"$\langle N\rangle ({0})$".format(mhlabel))
    save_plot(fig, "occupation", sim)
    return


def plot_vpec():
    fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
    bins = [np.linspace(0, 3000, 100), np.logspace(-1, 3.5, 100)]
    for ax, b in zip(axes, bins):
        ax.hist(subs.velocity(), b, histtype="step", label="3d", bottom=1)
        for i, x in enumerate("xyz"):
            ax.hist(subs.velocity(i), b, histtype="step", label=x, bottom=1)
        ax.legend(loc="upper right")
        ax.set_xlabel("Peculiar velocity (km/s)")
        ax.set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_title("Linear binning")
    axes[1].set_title("Log-scale binning")
    output = "vpeculiar"
    save_plot(fig, output, sim)


def average(sample, xbins, xname="stars", yname="total", mask=None, debug=False):
    if mask is None:
        mask = np.ones(sample.mass("total").size, dtype=bool)
    if debug:
        print("n =", np.histogram(sample.mass(xname), xbins)[0])
    yavg = np.histogram(
        sample.mass(xname)[mask], xbins, weights=sample.mass(yname)[mask]
    )[0]
    yavg = yavg / np.histogram(sample.mass(xname)[mask], xbins)[0]
    return yavg


def munari13(m, cosmo, z=0):
    hz = (cosmo.H(z) / (100 * u.km / u.s / u.Mpc)).value
    return 1177 * (hz * m / 1e15) ** 0.364


def test_velocities(subs):
    _, clusters = np.unique(subs.catalog["HostHaloId"], return_index=True)
    print("clusters =", clusters.min(), clusters.max(), clusters.shape)
    # print('Nbound[clusters] =', subs.catalog['Nbound'][clusters])
    fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
    ax = axes[0]
    # for proj in ('xyz', 'x', 'y', 'z'):
    # vi = np.array(subs.catalog['v_cl_{0}'.format(proj)])
    for proj in ("", "1", "2", "3"):
        vi = np.array(subs.catalog["Physical{1}HostMeanVelocity{0}".format(proj, vref)])
        print("vi =", vi.min(), vi.max(), np.percentile(vi, [1, 5, 25, 50, 75, 95, 99]))
        nanvi = ~np.isfinite(vi)
        print(
            "nan: {0}/{1} ({2:.1f}%)".format(
                nanvi.sum(), nanvi.size, 100 * nanvi.sum() / nanvi.size
            )
        )
        nanhosts = np.array(subs.catalog["HostHaloId"])[nanvi]
        goodhosts = np.array(subs.catalog["HostHaloId"])[~nanvi]
        vavgcols = ["PhysicalAverageVelocity{0}".format(i) for i in range(3)]
        vmbcols = ["PhysicalMostBoundVelocity{0}".format(i) for i in range(3)]
        ncols = ["NboundType{0}".format(i) for i in range(6)]
        mcols = ["MboundType{0}".format(i) for i in range(6)]
        print()
        print("in each nanhalo:")
        for hostid in nanhosts[:10]:
            inhalo = np.where(subs.catalog["HostHaloId"] == hostid)[0]
            print(
                "ID:",
                hostid,
                "| Nsub:",
                inhalo.size,
                "| Nbound > 0:",
                (subs.catalog["Nbound"][inhalo] > 0).sum(),
                "| NboundType > 0:",
                [(subs.catalog[c][inhalo] > 0).sum() for c in ncols],
                "| v_avg < inf:",
                [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vavgcols],
                "| v_bound < inf:",
                [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vmbcols],
            )
        print("in each good halo:")
        for hostid in goodhosts[:10]:
            inhalo = np.where(subs.catalog["HostHaloId"] == hostid)[0]
            print(
                "ID:",
                hostid,
                "Nsub:",
                inhalo.size,
                "| Nbound > 0:",
                (subs.catalog["Nbound"][inhalo] > 0).sum(),
                "| NboundType > 0:",
                [(subs.catalog[c][inhalo] > 0).sum() for c in ncols],
                "| v_avg < inf:",
                [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vavgcols],
                "| v_bound < inf:",
                [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vmbcols],
            )
        print()
        print("high-v:")
        highv = subs.catalog[vi > 1e8]
        # print(highv[['HostHaloId','Rank']])
        # print(highv['HostHaloId'].size, np.unique(highv['HostHaloId']))
        cols = [
            "HostHaloId",
            "PhysicalHostMeanVelocity",
            "PhysicalHostVelocityDispersion",
        ]
        mult = highv.groupby("HostHaloId")[cols].count()
        # print('mult =', mult)
        ax.hist(vi[clusters], 50, histtype="step", label=proj)
    ax.set_xlabel("Cluster velocity")
    ax.set_ylabel(r"$N_\mathrm{cl}$")
    ax = axes[1]
    for i, vi in enumerate(v, 1):
        ax.hist(vi, 100, histtype="step", label=str(i))
    ax.set_xlabel("Subhalo velocity")
    ax.set_ylabel(r"$N_\mathrm{sub}$")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=15)
    out = os.path.join(sim.plot_path, "hist_velocities.pdf")
    savefig(out, fig=fig)
    return


def view_velocities(subs, sim, vref="MostBound"):
    # note that I just want to look at hosts here
    cat = np.array(subs.catalog["Rank"] == 0)
    vcol = "Physical{0}HostMeanVelocity".format(vref)
    scol = "Physical{0}HostVelocityDispersion".format(vref)
    vinf = ~np.isfinite(subs.catalog[vcol])[cat]
    sinf = ~np.isfinite(subs.catalog[scol])[cat]
    logm = np.log10(subs.mass("total"))[cat]
    logmstar = np.log10(subs.mass("stars"))[cat]
    logmbins = np.arange(6, 16.01, 0.2)
    # histogram of velocities
    fig, axes = plt.subplots(figsize=(15, 6), ncols=2)
    ax = axes[0]
    for ax, m in zip(axes, (logm, logmstar)):
        ax.hist(
            m, logmbins, histtype="stepfilled", label="all", bottom=1, lw=0, alpha=0.5
        )
        ax.hist(
            m[vinf],
            logmbins,
            histtype="step",
            bottom=1,
            lw=2,
            label=r"$\langle v \rangle = \infty$",
        )
        ax.hist(
            m[sinf],
            logmbins,
            histtype="step",
            bottom=1,
            lw=2,
            label=r"$\sigma_\mathrm{v} = \infty$",
        )
    axes[0].set_xlabel(r"log $m_\mathrm{sub}$")
    axes[1].set_xlabel(r"log $m_\star$")
    for ax in axes:
        ax.set_yscale("log")
        ylim = ax.get_ylim()
        ax.set_ylim(1, ylim[1])
        ax.set_ylabel("1+N")
        ax.legend()
    out = os.path.join(sim.plot_path, "vinf.pdf")
    savefig(out, fig=fig)
    return


def output(sim, xname, yname):
    return "{0}_{1}".format(
        sim.masslabel(mtype=xname, latex=False), sim.masslabel(mtype=yname, latex=False)
    )


main()
