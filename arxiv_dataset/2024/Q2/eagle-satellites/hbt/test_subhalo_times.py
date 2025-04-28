from glob import glob
import h5py
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from time import time
from tqdm import tqdm
import sys

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.helpers.plot_definitions import binlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track


def main():
    args = hbt_tools.parse_args()
    if not args.debug:
        ic.disable()

    sim = Simulation(args.simulation, args.root)
    history_filename = os.path.join(sim.data_path, "history", "history.h5")
    if not os.path.isfile(history_filename):
        raise OSError(f"history file {history_filename} does not exist")
    ic(f"Loaded {sim.name} with {sim.snapshots.size} snapshots!")

    to = time()
    reader = HBTReader(sim.path)
    print("Loaded reader in {0:.1f} seconds".format(time() - to))

    isnap = -1
    ti = time()
    subs = reader.LoadSubhalos(isnap)
    print(f"Loaded subhalos in {time()-ti:.2f} s!")

    subs = Subhalos(
        subs,
        sim,
        isnap,
        exclude_non_FoF=True,
        load_distances=False,
        load_velocities=False,
    )
    subhalos = subs.catalog
    sats = subs.satellites
    cents = subs.centrals

    # load history

    # choose a few tracks
    rng = np.random.default_rng(31)
    sample_ids = rng.choice(sats["TrackId"], size=4)
    ic(sample_ids)

    fig, ax = plt.subplots()
    for i, trackid in enumerate(sample_ids):
        # if check_infall(trackid, subs, history_filename):
        # break
        track = Track(trackid, sim)
        ic(track)
        sub = subs.catalog[subs.catalog["TrackId"] == trackid]
        plot_mstar(ax, track, sub, color=f"C{i}")
        # break
    ax.legend(fontsize=12)
    ax.set(xlabel="Lookback time (Gyr)", ylabel=f"${binlabel['Mstar']}$", yscale="log")
    hbt_tools.save_plot(fig, "max_Mstar", track.sim)

    return


def plot_mstar(ax, track, sub, color="C0"):
    mstar = track["Mstar"]
    ic(mstar)
    ic(sub["history:max_Mstar:time"])
    ax.plot(track.sim.t_lookback, mstar, f"{color}-")
    j = np.argmax(mstar)
    ax.plot(track.sim.t_lookback[j], mstar[j], "o", mfc="w", mec=color, mew=2)
    ax.plot(
        sub["history:max_Mstar:time"],
        sub["history:max_Mstar:Mstar"],
        f"{color}*",
        ms=12,
        label=track.trackid,
    )
    # ax.annotate(
    #    track.trackid, xy=(0.95,0.95), xycoords='axes fraction',
    #    ha='right', va='top', fontsize=14)
    return


def check_infall(trackid, subs, history_filename):
    ic()
    ic(trackid, type(trackid))
    # obj_idx = subs.catalog['TrackId'] == trackid
    # ic(obj_idx.shape, obj_idx.sum())
    ic()
    with h5py.File(history_filename, "r") as hdf:
        trackids = np.array(hdf.get("trackids/TrackId"))
        j = trackids == trackid
        ic(hdf["trackids/TrackId"][j])
        ic(hdf["trackids/TrackId_current_host"][j])
        ic(hdf["trackids/TrackId_previous_host"][j])
        ic(hdf["last_infall/isnap"][j], hdf["last_infall/z"][j])
        ic(hdf["first_infall/isnap"][j], hdf["first_infall/z"][j])
        ic(hdf["sat/isnap"][j], hdf["sat/z"][j])
    ic()
    track = Track(trackid, subs.sim)
    print(track)
    ic(track.first_satellite_snapshot_index, track.last_central_snapshot)
    # infall_index = track.infall()
    # ic(infall_index)
    # this just to make sure
    infall_index_brute = track.infall(min_snap_range_brute=400)
    ic(infall_index_brute)
    print()
    return True


if __name__ == "__main__":
    main()
