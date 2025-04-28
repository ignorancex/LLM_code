from inspect import signature
from matplotlib import pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import (
    binned_statistic as binstat,
    binned_statistic_dd as binstat_dd,
    pearsonr,
)
from time import time
from tqdm import tqdm

import plottery
from plottery.plotutils import colorscale, savefig, update_rcParams

update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot
from hbtpy.helpers.plot_auxiliaries import (
    binning,
    definitions,
    get_axlabel,
    get_bins,
    get_label_bincol,
    logcenters,
    massbins,
    plot_line,
)
from hbtpy.helpers.plot_definitions import (
    ccolor,
    scolor,
    massnames,
    units,
    xbins,
    binlabel,
    events,
    axlabel,
)
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)
    ic(sim.path)
    ic(sim.virial_snapshots)

    reader = HBTReader(sim.path)

    isnap = 365
    subs0 = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logM200Mean_min=13,
        verbose_when_loading=False,
    )
    events = ("cent", "sat", "first_infall", "last_infall")
    ic(subs0.size)
    isnan = [
        np.arange(subs0.size)[np.isnan(subs0[f"history:{e}:isnap"])] for e in events
    ]
    ic(isnan)
    for event in events:
        ic(np.isnan(subs0[f"history:{event}:isnap"]).sum())
        ic(np.isnan(subs0["SnapshotIndexOfLastIsolation"]).sum())
        good = np.isfinite(subs0[f"history:{event}:isnap"]) & np.isfinite(
            subs0["SnapshotIndexOfLastIsolation"]
        )
        r = pearsonr(
            subs0[f"history:{event}:isnap"][good],
            subs0["SnapshotIndexOfLastIsolation"][good],
        )
        print(event, r)
    # for event in events:
    return

    snaps = np.array(
        [
            i
            for i in sim.snapshots[:-1]
            if ((subs0["SnapshotIndexOfLastIsolation"] == i).sum() > 0)
        ],
        dtype=int,
    )
    snaps = sim.virial_snapshots[np.in1d(sim.virial_snapshots, snaps)]
    print(f"{len(snaps)} relevant snapshots")
    # run!
    results = []
    if args.ncores == 1:
        results = [
            match_snapshot(args, reader, subs0, isnap, index=i)
            for i, isnap in tqdm(enumerate(snaps), total=len(snaps))
        ]
    else:
        pool = mp.Pool(args.ncores)
        for i, isnap in tqdm(enumerate(snaps), total=len(snaps)):
            out = pool.apply_async(
                match_snapshot,
                args=(args, reader, subs0, isnap),
                kwds={"index": i, "verbose": True},
            )
            results.append(out.get())
        pool.close()
        pool.join()
    # recover output
    nsnaps = snaps.size
    returned = np.zeros(nsnaps, dtype=bool)
    iso_mbound, iso_mstar, iso_mratio = np.zeros((3, nsnaps))
    segcurve_signature = signature(segregation_curve)
    nparams = len(segcurve_signature.params) - 1
    segfit = np.zeros((nsnaps, nparams))
    segcov = np.zeros((nsnaps, nparams, nparams))
    for out in results:
        if out is None:
            continue
        i = out["index"]
        returned[i] = True
        iso_mbound[i], iso_mstar[i], iso_mratio[i] = out["isolation masses"]
        segfit[i], segcov[i] = out["segregation fit"]

    store_segregation_fits(args, sim, snaps, segfit, segcov, verbose=True)

    store_mean_mass(
        args,
        sim,
        snaps[returned],
        iso_mbound[returned],
        iso_mstar[returned],
        iso_mratio[returned],
    )
    return


###############################################################
###############################################################
##
## Per-snapshot work
##
###############################################################
###############################################################


def fit_segregation(subs, at_iso):
    R = subs["ComovingMostBoundDistance/R200MeanComoving"][at_iso]
    hsmr = subs["Mbound/Mstar"][at_iso]
    mstar = subs["Mstar"][at_iso]
    fit, cov = curve_fit(
        segregation_curve, [np.log10(R), np.log(mstar)], np.log10(hsmr)
    )
    return fit, cov


def segregation_curve(x, a, b, c):
    logR, logmstar = x
    return a + b * logR + c * logmstar


def get_masses(subs, at_iso):
    iso_mbound = subs["Mbound"][at_iso].mean()
    iso_mstar = subs["Mstar"][at_iso].mean()
    iso_mratio = np.nanmean(subs["Mbound/Mstar"][at_iso])
    return iso_mbound, iso_mstar, iso_mratio


def match_snapshot(args, reader, subs0, isnap, index=None, verbose=False):
    iso = subs0["SnapshotIndexOfLastIsolation"] == isnap
    if iso.sum() == 0:
        return
    ti = time()
    if index is None:
        index = isnap
    # this stores all the outputs
    result = {"index": index}
    # load sample
    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        subs0.sim,
        isnap,
        logMmin=0,
        load_distances=True,
        load_velocities=False,
        load_history=False,
        verbose_when_loading=False,
    )
    at_iso = np.in1d(subs["TrackId"], subs0["TrackId"][iso])
    print("at_iso =", at_iso.sum())
    # masses at last isolation
    result["isolation masses"] = get_masses(subs, at_iso)
    # segregation relation
    result["segregation fit"] = fit_segregation(subs, at_iso)
    if verbose:
        print(f"{isnap}: matched {at_iso.sum()} subhaloes in {time()-ti:.2f}s")
    return result


###############################################################
###############################################################
##
## Store/plot results
##
###############################################################
###############################################################


def store_mean_mass(args, sim, snaps, iso_mbound, iso_mstar, iso_mratio, verbose=True):
    outfile = "segregation/masses_at_last_isolation"
    output = os.path.join(sim.data_path, f"{outfile}.txt")
    np.savetxt(
        output,
        np.transpose([snaps, iso_mbound, iso_mstar, iso_mratio]),
        fmt="%3d %.2e %.2e %.2e",
        header="isnap Mbound Mstar Mbound/Mstar",
    )
    print(f"Saved to {output}")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    for ax, mx in zip(axes, (iso_mbound, iso_mstar, iso_mratio)):
        ax.plot(snaps, mx, "o-", lw=3)
        ax.set(yscale="log")
    axes[0].set_ylabel(r"$\langle m_\mathrm{sub}^\mathrm{isol}\rangle$ (M$_\odot$)")
    axes[1].set_ylabel(
        r"$\langle$" + "$m_\mathrm{\u2605}^\mathrm{isol}$" + r"$\rangle$ (M$_\odot$)"
    )
    axes[2].set_ylabel(
        r"$\langle m_\mathrm{sub}^\mathrm{isol}/$"
        + "$m_\mathrm{\u2605}^\mathrm{isol}$"
        + r"$\rangle$"
    )
    axes[2].set_xlabel("Snapshot")
    for ax in axes[:-1]:
        ax.set(xticklabels=[])
    save_plot(fig, outfile, sim, tight=False)
    return


def store_segregation_fits(args, sim, snaps, segfit, segcov, verbose=True):
    outfile = "segregation/segregation_fits"
    output = os.path.join(sim.data_path, f"{outfile}.txt")
    nparams = segfit.shape[1]
    fit_fmt = " ".join(["%.2e" for i in segfit[0]])
    segcov_1d = np.array([i[np.triu_indices_from(i)] for i in segcov])
    cov_fmt = " ".join(["%.4e" for i in segcov_1d[0]])
    np.savetxt(
        output,
        np.vstack([snaps, segfit.T, segcov_1d.T]),
        fmt=f"%3d {fit_fmt} {cov_fmt}",
    )
    print(f"Saved to {output}")
    # plot
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    ax.plot(*segfit[:2], "o")
    ax.set(xlabel="a", ylabel="b")
    save_plot(fig, outfile, sim, tight=False)
    return


main()
