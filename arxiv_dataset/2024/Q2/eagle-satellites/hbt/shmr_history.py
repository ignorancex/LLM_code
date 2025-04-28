from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.collections import LineCollection
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import sys
from time import time
from tqdm import tqdm
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams

update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track, HaloTracks


warnings.simplefilter("ignore", RuntimeWarning)


def main():
    checkpoints = ["birth", "sat", "cent", "first_infall", "last_infall", "today"]
    # starts = _checkpoints[:-1]
    # ends = _checkpoints[1:]
    # if len(args.checkpoints) == 0:
    #     checkpoints = [(s, e) for i, e in enumerate(ends, 1) for s in starts[:i]]
    # else:
    #     checkpoints = [(start,end) for start, end
    #                    in zip(args.checkpoints[:-1], args.checkpoints[1:])]
    args = hbt_tools.parse_args(
        args=[
            ("--checkpoints", {"nargs": "*", "default": checkpoints}),
            ("--halo", {"default": None, "type": int}),
            ("--maxmass", {"default": 1e16, "type": float}),
            ("--minmass", {"default": 5e13, "type": float}),
            ("--nhalos", {"default": 0, "type": int}),
            ("--nsub", {"default": 50, "type": int}),
            ("--seed", {"default": 31, "type": int}),
        ]
    )
    np.random.seed(args.seed)
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)

    isnap = -1
    # min masses are the same as in mass_relations.py
    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logMmin=None,
        logM200Mean_min=9,
        logMstar_min=8,
    )
    if "sat" in args.checkpoints:
        mask = subs.central_mask | (subs["history:sat:time"] < 12)
        ic(mask.sum())
        subs = Subhalos(
            subs[mask],
            sim,
            isnap,
            logMmin=None,
            logM200Mean_min=None,
            logMstar_min=None,
            load_any=False,
        )
    print("Loaded subhalos!")

    shmr_history(args, subs)
    return


def shmr_history(args, subs):
    sats = subs.satellites
    cens = subs.centrals
    jsort = np.argsort(cens["M200Mean"]).values
    ic(cens["M200Mean"].iloc[jsort])
    if args.test:
        j = jsort[-5]
        halo_shmr_evolution(
            args,
            subs,
            subs.sim,
            cens["HostHaloId"].iloc[j],
            n=10,
            seed=66,
            hostmass=cens["M200Mean"].iloc[j],
        )
        return
    if args.halo is not None:
        j = (cens["TrackId"].iloc[jsort] == args.halo).values
        jarr = jsort[j]
    else:
        j = (cens["M200Mean"].iloc[jsort] > args.minmass) & (
            cens["M200Mean"].iloc[jsort] <= args.maxmass
        )
        jarr = jsort[j]
        ic(j.size, j.sum(), jarr)
        if args.nhalos > 0:
            jarr = np.random.choice(jarr, args.nhalos, replace=False)
    print(
        f"Total {jarr.size} halos with {np.log10(args.minmass):.2f}"
        f" < log M200Mean <= {np.log10(args.maxmass):.2f}"
    )
    for j in jarr:
        print()
        mhost = cens["M200Mean"].iloc[j]
        trackid = cens["TrackId"].iloc[j]
        hostid = cens["HostHaloId"].iloc[j]
        print(
            f"HostHaloID {hostid} / TrackID {trackid} (logM/Msun={np.log10(mhost):.2f})"
        )
        halo_shmr_evolution(
            args,
            subs,
            subs.sim,
            hostid,
            n=args.nsub,
            ncores=args.ncores,
            hostmass=mhost,
        )
    return


def halo_shmr_evolution(
    args,
    subs,
    sim,
    host_halo_id,
    seed=None,
    n=30,
    cmap="inferno_r",
    vmin=0,
    vmax=13.8,
    ncores=10,
    hostmass=None,
    show_shsmr=True,
    show_chsmr=True,
    show_central=True,
    draw_constant_ratios=True,
):
    ic()
    # this is used to register expected order of things
    assert (n > 0) or (
        seed is not None and seed >= 0
    ), "either `n` or `seed` must be specified and greater than zero"
    ic(host_halo_id)
    np.random.seed(seed)
    hhid = subs["HostHaloId"]
    hhid_mask = subs["HostHaloId"] == host_halo_id
    ic(hhid_mask)
    ic(hhid_mask.sum())
    halo = subs[hhid_mask]
    ic(halo.shape)
    c = halo["TrackId"].loc[halo["Rank"] == 0].iloc[0]
    ic(c)
    # haloid =
    cent = Track(c, sim)
    # ic(cent)
    sats = halo.loc[halo["Rank"] > 0]
    # vmin, vmax so the color scale is consistent
    if vmin is None:
        vmin = sim.cosmology.age(sim.redshifts[0]).value
    if vmax is None:
        vmax = sim.cosmology.age(sim.redshifts[-1]).value
    ic(vmin, vmax)
    # output = define_output(args, halo, start, end)
    # ic(output)
    jsat = select_satellites(args, sats["MboundType4"].values)
    nsat = sats["MboundType4"] > 0
    print(f"{c}: Selected {jsat.size}/{nsat.sum()} satellites")
    ic(jsat)
    ti = time()
    load_kwargs = dict(vmin=vmin, vmax=vmax, checkpoints=args.checkpoints, cmap=cmap)
    if ncores > 1:
        with Pool(ncores) as pool:
            out = [
                pool.apply_async(
                    load_shmr_track,
                    args=(sats["TrackId"].iloc[j], sim),
                    kwds=load_kwargs,
                )
                for j in jsat
            ]
            pool.close()
            pool.join()
        xy = [i.get() for i in out]
    else:
        xy = []
        for i, j in enumerate(jsat):
            if i % 10 == 0:
                print(f"i={i:2d}, jsat={j:5d}...")
            trackid = sats["TrackId"].iloc[j]
            xy.append(load_shmr_track(trackid, sim, **load_kwargs))
    xy = [xy_i for xy_i in xy if xy_i is not None]  # and xy_i[0] > 0]
    ic(len(xy), len(xy[0]), len(xy[0][0]))
    if show_central:
        load_kwargs["checkpoints"] = ("birth", "today")
        # the [0] is needed because load_shmr_track now always returns an array of histories
        xy_c = load_shmr_track(c, sim, **load_kwargs)[0]
    else:
        xy_c = None
    ic(len(xy_c))
    print(f"{c}: Loaded tracks in {time()-ti:.2f} s")
    # plot preamble
    colors, cmap = colorscale(n=cent.z.size, vmin=vmin, vmax=vmax, cmap=cmap)
    annot_kwargs = dict(
        xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top", fontsize=18
    )
    if hostmass is not None:
        logm = np.log10(hostmass)
        mlabel = rf"$\log\,M_\mathrm{{200m}}/\mathrm{{M}}_\odot = {logm:.2f}$"
    ylim = (3e9, 2e14) if show_central else (3e9, 1e14)
    ncols = len(xy[0])
    ic(ncols)
    # plot
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.5 * ncols, 6), constrained_layout=True
    )
    # for i, (ax, xy_i) in enumerate(zip(axes, xy)):
    for i, ax in enumerate(axes):
        # plot overall SHMR for reference
        if show_chsmr:
            kw = dict(
                sample="centrals",
                ls="-.",
                color="k",
                lw=1,
                hostmass="M200Mean",
                zorder=10,
                bins=np.logspace(8, 12, 9),
                min_hostmass=1e9,
            )
            subs.hsmr(ax=ax, **kw)
        if show_shsmr:
            kw = dict(
                sample="satellites",
                ls="-",
                color="k",
                lw=1,
                hostmass="M200Mean",
                zorder=10,
                bins=np.logspace(8, 12, 9),
                min_hostmass=1e13,
            )
            subs.hsmr(ax=ax, **kw)
        # central subhalo
        if xy_c is not None:
            plot_shmr_track(ax, *xy_c, is_central=True, size=10)
        # plot subhalos
        _ = [
            plot_shmr_track(ax, xy_i[i][0], xy_i[i][1], colors=xy_i[i][2])
            for xy_i in xy
        ]
        ic(args.checkpoints, i, args.checkpoints[i : i + 2])
        start, end = [chkp.replace("_", " ") for chkp in args.checkpoints[i : i + 2]]
        text = rf"{start} $\rightarrow$ {end}"
        if hostmass is not None:
            text = "\n".join([mlabel, text])
        ax.annotate(text, **annot_kwargs)
        ax.set(
            xlabel="$m_{\u2605}$" + r" $(\mathrm{M}_\odot)$",
            xlim=(1e8, 1.5e12),
            ylim=ylim,
            xscale="log",
            yscale="log",
        )
        ax.tick_params(axis="x", which="major", pad=10)
    axes[0].set(ylabel=r"$m_\mathrm{sub}\,(\mathrm{M}_\odot)$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].annotate(
        f"HostHaloId={host_halo_id}",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=14,
        fontfamily="monospace",
    )
    if draw_constant_ratios:
        rc = "0.5"
        x = np.logspace(8, 12.3, 10)
        for r in np.logspace(0, 3, 4):
            if r < 0:
                msg = rf"$m_\mathrm{{sub}}/" + "m_\u2605" + f"={r:.1f}$"
            else:
                msg = rf"$m_\mathrm{{sub}}/" + "m_\u2605" + f"={r:.0f}$"
            for ax in axes:
                line = ax.plot(x, r * x, rc, lw=1, dashes=(6, 4))
                if r in (1, 1000):
                    ax.text(
                        5e11 / r**0.5,
                        0.70 * r**0.5 * 5e11,
                        msg,
                        va="top",
                        ha="right",
                        color=rc,
                        rotation=40,
                        fontsize=12,
                    )
    # for ax in (ax_infall, ax_present):
    plt.colorbar(cmap, ax=axes, label="Universe age (Gyr)", pad=0.01)
    output = define_output(args, halo)
    save_plot(fig, output, sim, tight=False)
    return


def load_shmr_track(
    trackid,
    sim,
    mkey="Mbound",
    vmin=0,
    vmax=14,
    checkpoints=("birth", "today"),
    cmap="viridis",
):
    """
    checkpoints must any set of valid checkpoint:
        ('birth', 'cent', 'sat', 'first_infall', 'last_infall', 'today'))
    """
    # note that in the current implementation it is possible to have
    # t_cent > t_sat for a given subhalo
    assert (
        not np.in1d(checkpoints, ["cent", "sat"]).sum() == 2
    ), f"ambiguous checkpoint set (cent,sat)"
    track = Track(trackid, sim)
    colors, _ = colorscale(n=track.z.size, vmin=vmin, vmax=vmax, cmap=cmap)
    if mkey == "Mbound":
        mass = track.Mbound
    elif mkey == "M200Mean":
        mass = track.track["M200Mean"]
    x = track.MboundType[:, 4]
    y = mass
    isnaps = []
    for i, chkp in enumerate(checkpoints):
        if chkp == "birth":
            isnaps.append(0)
            continue
        if chkp == "today":
            isnaps.append(-1)
            continue
        i_ = track.history(cols="isnap", when=chkp, raise_error=False)
        isnaps.append(-1 if i_ is None else i_)
        if i == 0:
            continue
        if isnaps[-1] <= isnaps[-2]:
            isnaps[-1] = isnaps[-2] + 1
    ic(trackid, checkpoints, isnaps)
    data = []
    for i, isnap in enumerate(isnaps[1:]):
        rng = np.s_[isnaps[i] : isnap]
        data.append([x[rng], y[rng], colors[rng]])
    return data


def plot_shmr_track(ax, x, y, colors="k", size=6, scatter_kwargs={}, is_central=False):
    segments = np.transpose(
        [np.array([x[:-1], x[1:]]), np.array([y[:-1], y[1:]])], axes=(2, 1, 0)
    )
    if len(colors) == 0:
        return
    lines = LineCollection(segments, colors=colors[1:])
    ax.add_collection(lines)
    ls = "*" if is_central else "o"
    # white border for clarity
    ax.plot(x[-1], y[-1], ls, mec="w", mew=1, ms=size + 1, zorder=99)
    ax.plot(
        x[-1],
        y[-1],
        ls,
        mec="k",
        mew=1,
        ms=size,
        mfc="w" if is_central else colors[-1],
        zorder=100,
    )
    return


def define_output(args, halo):
    mass_bins = (13, 13.3, 13.7, 14, 14.3, 15)
    c = halo["TrackId"].loc[halo["Rank"] == 0].iloc[0]
    logm = np.log10(halo["M200Mean"].iloc[0])
    logmlabel = f"{logm:.2f}".replace(".", "p")
    jbin = np.digitize(logm, mass_bins)
    mbins_label = [f"{i:.1f}".replace(".", "p") for i in mass_bins[jbin - 1 : jbin + 1]]
    suff = "__".join(args.checkpoints)
    output = os.path.join(
        "shmr_history",
        f"logm_{mbins_label[0]}_{mbins_label[1]}",
        f"shmr_track{c}__{suff}",
    )
    return output


def select_satellites(args, Mstar, nbins=10, logmstar_min=8.5):
    # selecting satellites this way allows for good coverage in mass,
    # randomly
    ic()
    jmass = np.argsort(Mstar)
    logm = np.log10(Mstar[jmass])
    good = np.isfinite(logm)
    ic(good.sum())
    if good.sum() < args.nsub:
        return jmass[good]
    if nbins > args.nsub:
        nbins = args.nsub
    n = args.nsub // nbins
    ic(args.nsub, nbins, args.nsub // nbins)
    # the n most massive
    j = jmass[-n:]
    ic(j, logm[-n:])
    mbins = np.linspace(logmstar_min, logm[-n], nbins)
    ic(mbins)
    # add a while to get to the desired nsub
    for i_mbin in range(len(mbins)):
        ji = (logm > mbins[i_mbin - 1]) & (logm <= mbins[i_mbin])
        ic(i_mbin, ji.sum())
        if ji.sum() == 0:
            continue
        ji = (
            np.random.choice(jmass[ji], n, replace=False) if ji.sum() > n else jmass[ji]
        )
        ic(mbins[i_mbin - 1 : i_mbin + 1], np.log10(Mstar)[ji])
        ic(ji)
        j = np.append(j, ji)
    ic(j)
    ic(np.sort(np.log10(Mstar[j])))
    ic(j.size)
    return j


main()
