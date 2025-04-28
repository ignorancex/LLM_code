#!/usr/bin/python3
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii, fits
from astropy.units import Quantity
import cmasher as cmr
from glob import glob
from icecream import ic
from itertools import count
from matplotlib import cm, colors as mplcolors, pyplot as plt, ticker, rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from mpl_toolkits.axes_grid1 import ImageGrid
import multiprocessing as mp
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d, pearsonr
import sys
from time import sleep, time

from plottery.plotutils import colorscale, savefig, update_rcParams

update_rcParams()
rcParams["text.latex.preamble"] += r",\usepackage{color}"

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.helpers.plot_definitions import axlabel, binlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos  # , Track
from hbtpy.track import Track

adjust_kwargs = dict(
    left=0.10, right=0.95, bottom=0.05, top=0.98, wspace=0.3, hspace=0.1
)


def main():
    print("Running...")
    args = (
        (
            "--demographics",
            {"action": "store_true"},
        ),
        ("--investigate", {"action": "store_true"}),
    )
    args = hbt_tools.parse_args(args=args)
    sim = Simulation(args.simulation, args.root)

    to = time()
    reader = HBTReader(sim.path)
    print(f"Loaded reader in {time()-to:.1f} seconds")
    to = time()
    subs = Subhalos(
        reader.LoadSubhalos(-1),
        sim,
        -1,
        as_dataframe=True,
        logMmin=9,
        logM200Mean_min=9,
    )
    print(np.median(subs["history:max_Mstar:time"][subs.satellite_mask]))

    print(f"Loaded subhalos in {(time()-to)/60:.2f} minutes")

    print(
        "In total there are {0} central and {1} satellite subhalos".format(
            subs.centrals.size, subs.satellites.size
        )
    )

    centrals = Subhalos(subs.centrals, sim, -1, load_any=False, logMstar_min=9)
    satellites = Subhalos(
        subs.satellites, sim, -1, load_any=False, logM200Mean_min=13, logMstar_min=9
    )
    print(np.median(satellites["history:max_Mstar:time"]))
    tdiff = satellites["history:max_Mbound:time"] - satellites["history:cent:time"]
    print((np.abs(tdiff) > 1).sum(), tdiff.size, (np.abs(tdiff) > 1).sum() / tdiff.size)
    # return
    print(np.sort(satellites.colnames))

    # central infallers
    tbins = np.arange(0, 14.5, 2)
    ic(tbins)
    t = satellites["history:first_infall:time"]
    d = satellites["history:first_infall:Depth"]
    nt = np.histogram(t, tbins)[0]
    ic(nt)
    ntc = np.histogram(t[d == 0], tbins)[0]
    ic(ntc, ntc / nt)
    tmb = satellites["history:max_Mbound:time"]
    late_mmax = np.histogram(t[tmb < t], tbins)[0]
    ic(late_mmax, late_mmax / nt)
    centrals_late_mmax = np.histogram(t[(d == 0) & (tmb < t)], tbins)[0]
    satellites_late_mmax = np.histogram(t[(d > 0) & (tmb < t)], tbins)[0]
    ic(centrals_late_mmax, centrals_late_mmax / late_mmax, centrals_late_mmax / ntc)
    ic(
        satellites_late_mmax,
        satellites_late_mmax / late_mmax,
        satellites_late_mmax / (nt - ntc),
    )
    return

    # fit HSMR
    # fit_hsmr(centrals)
    # fit_hsmr(satellites)

    if args.investigate:
        investigate_max_mstar(args, reader, satellites)
        print("\n*****\n")
        investigate_tsat(args, satellites)
        return

    # some quick demographics
    if args.demographics:
        demographics(satellites)
        return

    # sort host halos by mass
    print("Sorting by mass...")
    rank = {
        "Mbound": np.argsort(-centrals.Mbound).values,
        "M200Mean": np.argsort(-centrals.catalog["M200Mean"]).values,
    }
    # should not count on them being sorted by mass
    ids_cent = centrals["TrackId"]

    do_plot_centrals = False
    if do_plot_centrals:
        n = 20
        print("Plotting centrals...")
        to = time()
        title = "{1}: {0} most massive central subhalos".format(n, sim.formatted_name)
        plot_centrals(
            args,
            sim,
            reader,
            subs,
            subs.centrals,
            rank["Mbound"][:n],
            suffix=n,
            title=None,
        )
        # title=title)
        print("Finished plot_centrals in {0:.2f} minutes".format((time() - to) / 60))

    do_plot_halos = True
    if do_plot_halos:
        print()
        print("Plotting halos...")
        to = time()
        # total, gas, halo, stars
        # ['gas', 'DM', 'disk', 'bulge', 'stars', 'BH']
        ti = time()
        j = rank["M200Mean"][:3]
        ic(centrals.catalog[["TrackId", "Rank", "Mbound", "M200Mean"]].iloc[j])
        # sys.exit()
        suff = np.log10(np.median(centrals.catalog["Mbound"].iloc[j]))
        suff = "{0:.2f}".format(suff).replace(".", "p")
        plot_halos(args, sim, reader, subs, centrals.catalog["TrackId"].iloc[j], nsub=2)
        # print('Plotted mass #{0} in {1:.2f} minutes'.format(
        # massindex, (time()-to)/60))
        print("Finished plot_halos in {0:.2f} minutes".format((time() - to) / 60))

    return


def demographics(satellites):
    def demographic_stats(colname, mask=None, a=(99.5, 95, 90, 68, 50)):
        ic(colname)
        if mask is None:
            mask = np.ones(x.size, dtype=bool)
        if "time" in colname:
            ic(satellites["TrackId"][mask].values)
        x = satellites[colname][mask]
        print(
            f"  {colname} (N={mask.sum()}/{mask.size};"
            f" f={mask.sum()/mask.size:.3f})"
        )
        med = np.median(x)
        print(f"    median = {med:.2f}")
        for ai in a:
            p = np.percentile(x, [50 - ai / 2, 50 + ai / 2])
            print(f"    {ai}th range = {p[0]:.2e} - {p[1]:.2e}")
        # Latex format
        p = np.percentile(x, [25, 75, 5, 90])
        p = " & ".join(f"{pi:4.1f}" for pi in p)
        print(f"  {med:.2f} [] & {p}")
        return  # nothing for now

    times_out = "times_hist"
    fig_times, ax_times = plt.subplots(constrained_layout=True)
    tbins = np.arange(0, 14, 0.2)
    tx = (tbins[1:] + tbins[:-1]) / 2
    hist = {"times": tx}
    events = (
        "birth",
        "cent",
        "sat",
        "first_infall",
        "last_infall",
        "max_Mbound",
        "max_Mstar",
        "max_Mgas",
    )
    for event in events:
        print(event)
        h = f"history:{event}"
        good = satellites[f"{h}:time"] > -1
        ic(good.size, good.sum())
        hist[event] = np.array(
            np.histogram(satellites[f"{h}:time"][good], tbins)[0], dtype=float
        )
        hist[event][hist[event] == 0] = np.nan
        ax_times.plot(tx, hist[event], "-", lw=3, label=event)
        for t in ("time", "z"):
            demographic_stats(f"{h}:{t}", good)
        print()
    ax_times.legend(fontsize=14, facecolor="w", ncol=2)
    ax_times.set(xlabel="Lookback time (Gyr)", ylabel="N", yscale="log")
    ax_times.xaxis.set_major_locator(ticker.MultipleLocator(2))
    # ax_times.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    hbt_tools.save_plot(fig_times, times_out, satellites.sim, tight=False)
    output = os.path.join(satellites.sim.data_path, f"{times_out}.txt")
    hdr = f"# bin edges: {tbins} Gyr"
    formats = {key: "%.1f" if key == "times" else "%.0f" for key in hist}
    ascii.write(hist, output, format="fixed_width", formats=formats, overwrite=True)
    return


def investigate_max_mstar(args, reader, satellites, isnap=-2):
    ic(satellites.shape)
    sats_prior = reader.LoadSubhalos(isnap)
    sats_prior = Subhalos(
        sats_prior,
        satellites.sim,
        isnap,
        as_dataframe=True,
        logMmin=None,
        logMstar_min=None,
        logM200Mean_min=None,
        exclude_non_FoF=False,
        load_distances=False,
        load_velocities=False,
    )
    ic(np.sort(sats_prior.colnames))
    # return
    sats = satellites.merge(
        sats_prior, how="left", on="TrackId", suffixes=("", "_past")
    )
    grew = sats["Mstar"] > sats["MboundType4_past"]
    ic(grew.size, grew.sum())
    fig, ax = plt.subplots()
    ic(sats["IsDark"].sum(), sats["IsDark_past"].sum())
    good = ~(sats["IsDark"] | sats["IsDark_past"])
    ic(good.sum(), (1 - good).sum())
    ax.hist(np.log10(sats["Mstar"] / sats["MboundType4_past"])[good], 100, log=True)
    hbt_tools.save_plot(fig, "mstar_diff", sats.sim)
    return


def investigate_tsat(args, satellites):
    sim = satellites.sim
    tbins = np.arange(0, 13.5, 0.2)
    tsat = satellites["history:sat:time"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(tsat, tbins, histtype="stepfilled", color="C9")
    ax.set(xlabel="$t_\mathrm{sat}^\mathrm{lookback}$ (Gyr)")
    output = os.path.join(sim.plot_path, "early_tsat", "tsat_hist.png")
    early = tsat > 12
    events = (
        "birth",
        "cent",
        "first_infall",
        "last_infall",
        "max_Mbound",
        "max_Mstar",
        "max_Mgas",
    )

    def summary(x, label="", log=False):
        def _summary(x_, log=False):
            x0_ = np.median(x_)
            xlo, xhi = np.percentile(x_, [25, 75])
            if log:
                x0_, xlo, xhi = np.log10(np.array([x0_, xlo, xhi]))
            return x0_, xlo, xhi

        x0 = _summary(x[~early], log=log)
        xe = _summary(x[early], log=log)
        if label:
            print(f"{label}:")
        print(x0, xe)
        print(f"Early: {xe[0]:.3f} (50% range: {xe[1]:.3f}-{xe[2]:.3f}")
        print(f"Late:  {x0[0]:.3f} (50% range: {x0[1]:.3f}-{x0[2]:.3f}")

    for event in events:
        t = satellites[f"history:{event}:time"]
        print(f"--{event}--")
        summary(t, "times")
        summary(satellites[f"history:{event}:Mbound"], "msub", log=True)
        summary(satellites[f"history:{event}:Mstar"], "mstar", log=True)
    print("--Today--")
    summary(satellites["Mbound"], "msub", log=True)
    summary(satellites["Mstar"], "mstar", log=True)
    return


def fit_hsmr(subs, form="double power"):
    if form == "double power":
        f = lambda x, N, m1, b, g: 2 * N * x((m1 / x) ** b + (x / m1) ** g)


def member_indices(args, subcat, host_ids, nsub=10):
    cat = subcat.catalog
    centrals = cat["Rank"] == 0
    halos = np.intersect1d(cat["TrackId"], host_ids)
    to = time()
    if args.ncores == 1:
        # if True:
        idx = []
        for track in host_ids:
            idx.append(subcat.siblings(track, "index")[: nsub + 1])
    else:
        idx = [track for track in host_ids]
        ic(len(idx))
        pool = mp.Pool(args.ncores)
        results = [
            pool.apply_async(
                subcat.siblings, args=(track,), kwds={"return_value": "index"}
            )
            for track in host_ids
        ]
        pool.close()
        pool.join()
        for i, out in enumerate(results):
            out = out.get()
            try:
                idx[out[0]] = out[1]
            except IndexError as e:
                ic(i)
                raise IndexError(e)
    print("member indices in {0:.2f} s".format(time() - to))
    return idx


def plot_centrals(
    args,
    sim,
    reader,
    subcat,
    centrals,
    indices,
    massindex=-1,
    suffix=None,
    title="Central subhalos",
):
    # another option is to create more than one gridspec, see
    # https://matplotlib.org/users/tight_layout_guide.html
    # but I've wasted enough time here for now
    gridspec_kw = {"width_ratios": (1, 1, 0.05)}
    fig, (ax1, ax2, cax) = plt.subplots(
        figsize=(15, 6), ncols=3, gridspec_kw=gridspec_kw
    )
    axes = (ax1, ax2)
    if title:
        fig.suptitle(title)
    cscale, cmap = colorscale(np.log10(centrals.Mbound[indices]))
    track_data = read_and_plot_track(
        args,
        axes,
        sim,
        reader,
        subcat,
        centrals,
        indices,
        show_history=False,
        label_host=False,
        show_label=False,
        colors=cscale,
        label_history=False,
    )
    save_tracks(track_data, "track_centrals", sim, massindex, suffix)
    cbar = fig.colorbar(cmap, cax=cax)
    cbar.ax.tick_params(labelsize=20, direction="in", which="both", pad=14)
    clabel = sim.masslabel(mtype="total")
    cbar.set_label(r"log ${0}$ ($z=0$)".format(clabel))  # , fontsize=20)
    setup_track_axes(axes, sim, sim.cosmology)
    output = os.path.join(sim.plot_path, "track_centrals")
    if suffix:
        output += "_{0}".format(suffix)
    savefig("{0}.pdf".format(output), fig=fig, rect=[0, 0, 1, 0.95])
    return


def make_halo_plot(
    args,
    sim,
    tracks,
    massindex=-1,
    includes_central=True,
    title=None,
    label_history=True,
    suffix="",
):
    to = time()
    # this should include multiple clusters
    if not hasattr(tracks[0], "__iter__"):
        tracks = [tracks]
    ic(includes_central)
    i0 = int(includes_central)
    ic(i0)
    hostids = [t[-3] for t in tracks]
    ic(hostids)
    ncl = np.unique(hostids).size
    nsub = [(hostids == i).sum() for i in np.unique(hostids)]
    nsat = [n - includes_central for n in nsub]
    # colors. Centrals are black; loop colors for satellites
    colors = []
    j = 0
    n = 0
    for i in range(len(tracks)):
        if i == sum(nsub[:n]):
            j = 0
            n += 1
            if includes_central:
                colors.append("k")
                continue
        colors.append("C{0}".format(j % 10))
        j += 1
    mname = sim.masslabel(index=massindex, latex=False)
    output = "track_{0}".format(mname)
    if suffix:
        output += "_{0}".format(suffix)
    output = os.path.join(sim.plot_path, "{0}.pdf".format(output))
    # title = '{0}: {1} most massive halos'.format(sim.formatted_name, ncl)
    fig, axes = plt.subplots(figsize=(14, 5 * (ncl + 0.5)), ncols=2, nrows=ncl)
    if ncl == 1:
        axes = [axes]
    for i, row in enumerate(axes):
        ic(i)
        n = int(np.sum(nsub[:i]))
        # the central track
        if includes_central:
            plot_track(
                row,
                sim,
                tracks[n],
                massindex=massindex,
                color=colors[n],
                label_history="right",
            )
        j = [n + i0 + ii for ii in range(nsub[i] - i0)]
        _ = [
            plot_track(row, sim, tracks[ji], massindex=massindex, color=colors[ji])
            for ji in j
        ]
        # need to pass the mass definition used to sort halos here,
        # for the upper-left text
        setup_track_axes(
            row,
            sim,
            sim.cosmology,
            is_last_row=(i == len(axes) - 1),
            masslabel=sim.masslabel(index=massindex, latex=True),
        )
    rect = [0.03, 0.01, 0.99, 0.99]
    if title:
        fig.suptitle(title)
        rect[3] -= 0.12 / ncl
    savefig(output, fig=fig, rect=rect, w_pad=1.2)
    print("    in {0:.1f} s".format(time() - to))
    return


def plot_halos(
    args, sim, reader, subcat, ids_central, nsub=10, title=None, suffix=None
):
    """Plot the evolution of a few massive subhalos in a set of halos"""
    to = time()
    ncl = len(ids_central)
    idx = member_indices(args, subcat, ids_central, nsub=nsub)
    idx = np.reshape(idx, -1)
    # this is the expensive step
    tracks = read_tracks(args, sim, reader, subcat.catalog, idx, nsub=nsub)
    # ['gas', 'DM', 'disk', 'bulge', 'stars', 'BH']
    # disk and bulge mass not included in EAGLE nor Apostle; skipping
    # saves a minute or two
    for massindex in (-1, 0, 1, 4, 5):
        make_halo_plot(args, sim, tracks, massindex=massindex, title=title)
        save_tracks(tracks, "track_halos", sim, massindex, suffix)
    return


def plot_track(
    axes,
    sim,
    track_data,
    massindex=-1,
    show_history=True,
    label_host=True,
    show_label=True,
    color="k",
    label_history=False,
    verbose=True,
    **kwargs,
):
    """
    Make sure `massindex` is consistent with `read_track()`
    """
    trackid, t, Mt, rank, depth, icent, isat, iinf, hostid, th, Mh = track_data
    ic(trackid)
    ic(rank[-1])
    if hasattr(massindex, "__iter__"):
        pass
    else:
        Mt = Mt[massindex]
        Mh = Mh[massindex]
    Mo = Mt[-1]
    if show_label:
        label = "{0}: {1:.2f} ({2}/{3})".format(
            trackid, np.log10(Mt[-1]), depth[-1], hostid
        )
    else:
        label = "_none_"
    axes[0].plot(t, Mt, label=label, color=color, **kwargs)
    axes[1].plot(t, Mt / Mo, color=color, **kwargs)
    # ic()
    # ic(label_host)

    if show_history:
        for ax, m in zip(axes, [Mt, Mt / Mo]):
            if icent is not None:
                ax.plot(
                    t[icent],
                    m[icent],
                    "ws",
                    mec=color,
                    ms=12,
                    mew=1.5,
                    label=r"$t_\mathrm{cen}$",
                )
            if isat is not None:
                ax.plot(
                    t[isat],
                    m[isat],
                    "wo",
                    mec=color,
                    ms=12,
                    mew=1.5,
                    label=r"$t_\mathrm{sat}$",
                )
            else:
                ax.plot(
                    [], [], "wo", mec=color, ms=12, mew=1.5, label=r"$t_\mathrm{sat}$"
                )
            ax.plot(
                t[iinf],
                m[iinf],
                "o",
                ms=7,
                color=color,
                mew=1.5,
                label=r"$t_\mathrm{infall}$",
            )
        if label_history:
            lax = axes[["left", "right"].index(label_history)]
            lax.legend(loc="lower right", bbox_to_anchor=(0.97, 0.08))
    # the main host halo - just plot when looking at the central
    # subhalo
    if rank[-1] == 0 and show_history:
        # host_kwargs = dict(dashes=(6,6), color='k', lw=1.5)
        # axes[0].plot(th, Mh, **host_kwargs)
        # axes[1].plot(th, Mh/Mh[-1], **host_kwargs)
        # some information on the host halo
        if label_host:
            ic()
            text = r"log ${1}^\mathrm{{host}} = {0:.2f}$".format(
                np.log10(Mh[-1]), sim.masslabel(index=massindex, latex=True)
            )
            text += "\nHost ID = {0}".format(hostid)
            txt = axes[0].text(
                0.04,
                0.97,
                text,
                ha="left",
                va="top",
                fontsize=16,
                transform=axes[0].transAxes,
            )
            txt.set_bbox(dict(facecolor="w", edgecolor="w", alpha=0.5))
    return


def read_and_plot_track(
    args,
    axes,
    sim,
    reader,
    subhalos,
    cat,
    indices,
    massindex=-1,
    show_history=True,
    label_host=True,
    show_label=True,
    colors=None,
    color="k",
    label_history=False,
    verbose=True,
    **kwargs,
):
    if not hasattr(indices, "__iter__"):
        indices = [indices]
    read_kwargs = {"load_history": show_history, "verbose": verbose}
    if colors is None:
        colors = [color] * len(indices)
    if args.ncores == 1:
        track_data = [
            read_track(sim, reader, cat, i, **read_kwargs)[:-1] for i in indices
        ]
    else:
        track_data = [[] for _ in indices]
        pool = mp.Pool(args.ncores)
        # need to pass the indices this way so I can keep track of order
        results = [
            pool.apply_async(
                read_track, args=(sim, reader, cat[indices], i), kwds=read_kwargs
            )
            for i in range(len(indices))
        ]
        pool.close()
        pool.join()
        for out in results:
            out = out.get()
            track_data[out[-1]] = out[:-1]
    _ = [
        plot_track(
            axes,
            sim,
            data,
            massindex=massindex,
            color=color,
            label_host=label_host,
            show_history=show_history,
            label_history=label_history,
            zorder=indices.size + 1 - i,
        )
        for i, data, color in zip(count(), track_data, colors)
    ]
    return track_data


def read_tracks(args, sim, reader, cat, indices, nsub=10, sort_mass=True, **kwargs):
    """
    if `sort_mass==True`, then tracks are sorted per host by final
    total mass. Note that this assumes that tracks belonging to
    each host are all together one after the other (will generalize
    later)
    """
    if not hasattr(indices, "__iter__"):
        indices = [indices]
    to = time()
    if args.ncores == 1:
        tracks = [read_track(sim, reader, cat, i, **kwargs)[:-1] for i in indices]
    else:
        tracks = [[] for _ in indices]
        pool = mp.Pool(args.ncores)
        # need to pass the indices this way so I can keep track of order
        results = [
            pool.apply_async(
                read_track, args=(sim, reader, cat[indices], i), kwds=kwargs
            )
            for i in range(len(indices))
        ]
        pool.close()
        pool.join()
        for out in results:
            out = out.get()
            tracks[out[-1]] = out[:-1]
    print("Read all tracks in {0:.2f} minutes".format((time() - to) / 60))
    # read_track output (last value not recorded):
    # return trackid, t, Mt, rank, depth, icent, isat, iinf, \
    # hostid, th, Mh, track_idx
    to = time()
    if sort_mass:
        hostids = np.array([t[-3] for t in tracks])
        nh = np.arange(hostids.size, dtype=int)
        # this is the last value of the last Mt column
        Mtot_now = np.array([t[2][-1][-1] for t in tracks])
        for i in np.unique(hostids):
            mask = hostids == i
            aux = [tracks[m] for m in nh[mask]]
            jsort = np.argsort(-Mtot_now[mask])
            for j, k in enumerate(nh[mask]):
                tracks[k] = aux[jsort[j]]
    print("Sorted tracks in {0:.2f} s".format(time() - to))
    return tracks


def read_track(
    sim, reader, cat, track_idx, load_history=True, massindex=None, verbose=True
):
    # ic()
    trackid = cat["TrackId"][track_idx]
    ti = time()
    track = Track(trackid, sim)
    # ic(track)
    # sys.exit()
    if verbose:
        print(
            "Loaded track #{2} (TrackID {0}) in {1:.2f} minutes".format(
                trackid, (time() - ti) / 60, track_idx
            )
        )
    # this is what takes up all the time, especially the infall
    if load_history:
        # las moment it was a central
        icent = track.last_central_snapshot_index
        # first time it was a satellite
        isat = track.first_satellite_snapshot_index
        # infall
        iinf = track.infall("index")
    else:
        icent = isat = iinf = track._none_value
    host = track.host(-1, return_value="track")
    host = Track(host, sim)
    ic(track)
    ic(track.lookback_time)
    t = track.lookback_time()
    th = host.lookback_time()
    # ic(massindex)
    if massindex is None:
        Mt = [track.mass(index=i) for i in range(6)]
        Mt.append(track.mass(index=-1))
        Mt = np.array(Mt)
        Mh = [host.mass(index=i) for i in range(6)]
        Mh.append(host.mass(index=-1))
        Mh = np.array(Mh)
    else:
        Mt = track.mass(index=massindex)
        Mh = host.mass(index=massindex)
    # ic(Mt.shape)
    # ic(Mt[-1])
    # ic(Mh.shape)
    # ic(Mh[-1])
    # print('Mt =', Mt.shape)
    # print(
    rank = track.track["Rank"]
    depth = track.track["Depth"]
    hostid = host.trackid
    return trackid, t, Mt, rank, depth, icent, isat, iinf, hostid, th, Mh, track_idx


def print_halo(halo, mmin=10):
    print()
    print("HostHaloId {0}".format(halo["TrackId"]))
    print(halo["Mbound"], (halo["Rank"] == 0).sum())
    print(
        "has a total mass {0:.2e} Msun/h".format(
            1e10 * halo["Mbound"][halo["Rank"] == 0][0]
        )
    )
    print("and {0} subhalos (including the central),".format(halo["Rank"].size))
    print("with")
    print("    {0} disrupted subhalos".format((halo["Nbound"] == 1).sum()))
    for mmin in np.logspace(-2, 4, 7):
        print(
            "    {0} having Mbound > {1:.2e} Msun/h".format(
                (halo["Mbound"] > mmin).sum(), 1e10 * mmin
            )
        )
    return


def save_tracks(track_data, output, sim=None, massindex=None, suffix=None):
    """Save track information to file

    Parameters
    ----------
    track_data : list
        output(s) of one or more runs of `read_track`
    """
    if output[-5:] == ".fits":
        output = output[:-5]
    if suffix is not None:
        output += "_{0}".format(suffix)
    if sim is not None:
        if massindex is not None:
            output += "_{0}".format(sim.masslabel(index=massindex, latex=False))
        output = os.path.join(sim.data_path, output)
    output += ".fits"
    if not hasattr(track_data[0], "__iter__"):
        track_data = [track_data]
    ntracks = len(track_data)
    nval = len(track_data[0])
    nsnap = sim.snapshots.size
    # "transpose"
    track_data = [[t[i] for t in track_data] for i in range(nval)]
    # should record all mass types - all of them are part of
    # track_data so just need to separate them here. Right?
    farr = "{0}E".format(nsnap)
    names = [
        "TrackId",
        "t_lookback",
        "Mt",
        "Rank",
        "Depth",
        "i_cen",
        "i_sat",
        "i_infall",
        "TrackId_host",
        "t_lookback_host",
        "Mt_host",
    ]
    fmts = ["J", farr, farr, "I", "I", "I", "I", "I", "J", farr, farr]
    columns = []
    for name, fmt, col in zip(names, fmts, track_data):
        if name[:2] == "Mt":
            col = [np.transpose(c) for c in col]
            name = [
                name.replace("Mt", sim.masslabel(index=i, latex=False))
                for i in range(-1, 6)
            ]
        else:
            if not hasattr(col[0], "__iter__"):
                col = np.array([col])
            name = [name]
        for i, c, n in zip(count(), col, name):
            if isinstance(c, Quantity):
                c = c.value
            if np.any(c):
                columns.append(fits.Column(name=n, array=c, format=fmt))
    fitstbl = fits.BinTableHDU.from_columns(columns)
    if os.path.isfile(output):
        os.remove(output)
    fitstbl.writeto(output)
    print("Saved to", output)
    return


def summarize_array(array, name, **kwargs):
    print(
        "{0}: (min,max)=({1},{2}); size={3}".format(
            name, array.min(), array.max(), array.size
        ),
        **kwargs,
    )
    return


def title_axis(title, rows=10):
    """Title axis wrapper"""
    tax = plt.subplot2grid((rows, 1), (0, 0))
    tax.axis("off")
    tax.text(0.5, 0.5, title, ha="center", va="center")
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    return rows


def setup_track_axes(
    axes,
    sim,
    lookback_cosmo=False,
    zmarks=[0.1, 0.5, 1, 2, 5],
    textcolor="0.3",
    masslabel=r"M_\mathrm{total}",
    is_last_row=True,
):
    if isinstance(lookback_cosmo, FlatLambdaCDM):
        x = "t"
        xlabel = "lookback time (Gyr)"
        xlim = (14.5, -0.5)
        tickspace = 4
        tmarks = [lookback_cosmo.lookback_time(z).value for z in zmarks]
    else:
        x = "z"
        xlabel = "Redshift"
        xlim = (16, -0.5)
        tickspace = 5
    ylabel = [
        r"${1}({0})$".format(x, masslabel),
        r"${1}({0})/{1}$$_{{,0}}$".format(x, masslabel),
    ]
    axes[0].set_ylabel(ylabel[0])
    axes[1].set_ylabel(ylabel[1])
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tickspace))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ytext = get_ytext(ax)
        if is_last_row:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])
        if lookback_cosmo:
            for z, t in zip(zmarks, tmarks):
                ax.axvline(t, dashes=(4, 4), lw=1, color=textcolor)
                # in order to do this, need to know plot limits...
                ax.annotate(
                    "z={0:.1f}".format(z),
                    xy=(t - 0.1, ytext),
                    ha="left",
                    va="center",
                    fontsize=12,
                    color=textcolor,
                )
    return


def get_ytext(ax, height=0.05):
    ylim = ax.get_ylim()
    if ax.get_yscale() == "log":
        logy = np.log10(np.array(ylim))
        ytext = 10 ** (logy[0] + height * (logy[1] - logy[0]))
    else:
        ytext = ylim[0] + height * (ylim[1] - ylim[0])
    return ytext


main()
