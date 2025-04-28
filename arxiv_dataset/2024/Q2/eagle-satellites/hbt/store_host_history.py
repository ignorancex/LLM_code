import h5py
from icecream import ic
import numpy as np
import os
from scipy.interpolate import interp1d
from time import time
from tqdm import tqdm

from HBTReader import HBTReader

from hbtpy import hbt_tools
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track


def main():
    args = hbt_tools.parse_args(
        args=[("--no-store", {"dest": "store", "action": "store_false"})]
    )
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)
    isnap = -1
    # only the columns I need to identify hosts
    subs = reader.LoadSubhalos(isnap, ("HostHaloId", "TrackId", "Rank"))
    # here we load hosts only because we only need to store
    # host haloes above an M200Mean (just to make I/O faster)
    centrals = Subhalos(
        subs[subs["Rank"] == 0],
        sim,
        isnap,
        load_distances=False,
        load_velocities=False,
        load_history=False,
    )
    columns = np.sort(centrals.columns)
    # removing these columns I don't care about
    columns = columns[~np.in1d(columns, ["Rank"])]
    ic(columns)
    ic(centrals["TrackId"])
    ic(centrals.shape)
    ic(centrals.size)

    snaps = np.sort(sim.snapshots)
    # this will be stored in an hdf5 file
    host_history = {
        **{
            "snapshots": {
                "snapshots": snaps,
                "lookback": np.zeros(snaps.size),
                "z": np.zeros(snaps.size),
                "interpolated": np.ones(snaps.size, dtype=np.int8),
            },
            **{
                "TrackId": centrals["TrackId"],
                "HostHaloId": -np.ones((centrals.size, snaps.size), dtype=int),
            },
            **{
                key: -np.ones((centrals.size, snaps.size))
                for key in np.sort(centrals.columns)
                if key != "TrackId"
            },
        }
    }
    ic(host_history["M200Mean"].shape)
    hdf = os.path.join(sim.data_path, "history", "host_history.h5")

    snaps_with_host = (snaps < 115) | (snaps % 5 == 0)
    print("Reading host information...")
    host_history = load_host_history(
        args, reader, centrals, snaps, snaps_with_host, host_history, hdf
    )

    # now we loop again over missing snaps
    print("Interpolating missing snapshots...")
    host_history = interpolate_host_history(
        args, snaps, snaps_with_host, host_history, centrals.columns, hdf
    )

    if not args.store:
        write_h5(hdf, host_history, verbose=True)
    return


def interpolate_host_history(
    args, snaps, snaps_with_host, host_history, columns, hdf, write_freq=20
):
    t = host_history["snapshots"]["lookback"]
    rng = np.arange(snaps.size)
    for i, snap in enumerate(tqdm(snaps)):
        if snaps_with_host[i]:
            continue
        isnap_before = rng[snaps == (snap - snap % 5)][0]
        isnap_after = rng[snaps == (snap + (5 - snap % 5))][0]
        ic(snap, isnap_before, isnap_after)
        for key in columns:
            if key in ("HostHaloId", "TrackId"):
                continue
            y = [host_history[key][:, isnap_before], host_history[key][:, isnap_after]]
            ic(np.array(y).shape)
            func = interp1d([t[isnap_before], t[isnap_after]], y, kind="linear", axis=0)
            ic(func(t[isnap_before + 1 : isnap_after]).shape)
            host_history[key][:, isnap_before + 1 : isnap_after] = func(
                t[isnap_before + 1 : isnap_after]
            ).T
        if args.store and (i % write_freq == 0):
            write_h5(hdf, host_history)
        if args.test and snap > 140:
            break
    return host_history


def load_host_history(
    args, reader, centrals, snaps, snaps_with_host, host_history, hdf, write_freq=20
):
    z, t = np.zeros((2, snaps.size))
    for i, snap in enumerate(tqdm(snaps)):
        z[i] = centrals.sim.redshift(snap)
        t[i] = centrals.sim.cosmology.lookback_time(z[i]).to("Gyr").value
        for key, val in zip(("lookback", "z"), (t[i], z[i])):
            host_history["snapshots"][key][i] = val
        host_history["snapshots"]["interpolated"][i] = 0
        subs_i = reader.LoadSubhalos(snap)
        if subs_i.size == 0:
            continue
        centrals_i = Subhalos(
            subs_i[subs_i["Rank"] == 0],
            centrals.sim,
            snap,
            logMmin=None,
            logM200Mean_min=None,
            logMstar_min=None,
            load_distances=False,
            load_velocities=False,
            load_history=False,
            load_hosts=snaps_with_host[i],
            verbose_when_loading=False,
        )
        ic(i, snap, z[i])
        matched = centrals.merge(
            centrals_i, on="TrackId", how="left", suffixes=("_0", "")
        )
        matched = matched[centrals.columns]
        host_history["HostHaloId"][:, i] = matched["HostHaloId"].fillna(-99)
        if not snaps_with_host[i]:
            continue
        ic(matched.columns)
        ic(matched[["TrackId", "HaloId", "M200Mean"]])
        for col in matched:
            if col in ("HostHaloId", "TrackId"):
                continue
            host_history[col][:, i] = matched[col].fillna(-99)
        if args.store and (i % write_freq == 0):
            write_h5(hdf, host_history)
        if args.test and snap > 140:
            break
    return host_history


def write_h5(hdf, host_history, verbose=False):
    with h5py.File(hdf, "w") as out:
        gr = out.create_group("snapshots")
        for key, val in host_history["snapshots"].items():
            ic(key, val)
            gr.create_dataset(key, data=val, dtype=val.dtype)
        gr = out.create_group("hosts")
        for col, data in host_history.items():
            if col == "snapshots":
                continue
            gr.create_dataset(col, data=data)
    if verbose:
        print(f"Saved to {hdf}!")


main()
