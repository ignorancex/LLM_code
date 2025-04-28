import h5py
from icecream import IceCreamDebugger
import numpy as np
import os
import pandas as pd
from time import time
from tqdm import tqdm
import warnings

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

warnings.simplefilter("ignore", UserWarning)


icnames = ("birth", "cent", "sat", "infall", "history", "masses", "base")
ics = {name: IceCreamDebugger() for name in icnames}
ic0 = ics["base"]


def main():
    args = hbt_tools.parse_args(
        args=[("--no-store", {"dest": "store", "action": "store_false"})]
        + [(f"--{name}", {"action": "store_true"}) for name in icnames]
    )
    # individual ic instances
    for name in icnames:
        if not args.__dict__.get(name):
            ics[name].disable()
    if args.history:
        for name in icnames[:4]:
            ics[name].enable()
    if args.debug:
        ic0.enable()

    sim = Simulation(args.simulation, args.root)
    print(f"Loaded {sim.name} with {sim.snapshots.size} snapshots!")
    reader = HBTReader(sim.path)
    isnap = -1
    ti = time()
    subs = reader.LoadSubhalos(isnap)
    print(f"Loaded {subs.shape[0]} subhalos in {time()-ti:.2f} s!")

    subs = Subhalos(
        subs,
        sim,
        isnap,
        exclude_non_FoF=True,
        logMmin=None,
        logM200Mean_min=None,
        logMstar_min=None,
        load_distances=True,
        load_velocities=False,
        load_hosts=False,
        load_history=False,
    )
    subhalos = subs.catalog
    sats = subs.satellites
    cents = subs.centrals
    ic0(sats["TrackId"].shape)
    ic0(cents["TrackId"].shape)
    track_ids = subs.satellites["TrackId"]
    host_halo_id = sats["HostHaloId"].to_numpy(dtype=np.int32)
    ic0(host_halo_id, host_halo_id.shape)
    cent_track_id = cents["TrackId"].to_numpy(dtype=np.int32)
    ic(cent_track_id, cent_track_id.shape)
    ti = time()
    host_track_id_today = sats.merge(
        cents, on="HostHaloId", suffixes=("_sat", "_cent"), how="left"
    )["TrackId_cent"].values
    print(f"host tracks in {time()-ti:.2f} s")
    ic0(host_track_id_today[:5])

    groups = (
        "trackids",
        "max_Mbound",
        "max_Mstar",
        "max_Mdm",
        "max_Mgas",
        #   'max_Mstar/Mbound', 'max_Mdm/Mbound', 'max_Mgas/Mbound',
        #   'max_Mbound/Mstar', 'max_Mgas/Mstar',
        "birth",
        "first_infall",
        "last_infall",
        "sat",
        "cent",
        "peri",
    )
    names = [("TrackId", "TrackId_current_host", "TrackId_previous_host")] + (
        len(groups) - 1
    ) * [("isnap", "time", "z", "Mbound", "Mstar", "Mdm", "Mgas", "M200Mean", "Depth")]
    #           TrackIDs           isnap      time,z,masses      depth
    dtypes = [3 * [np.int32]] + (len(groups) - 1) * [
        [np.int16] + 7 * [np.float32] + [np.int8]
    ]
    cols = {
        group: [f"{group}:{name}" for name in grnames]
        for group, grnames in zip(groups, names)
    }
    # DataFrame
    history = {
        f"{gr}:{name}": -np.ones(subs.nsat, dtype=dt)
        for gr, grnames, dtype in zip(groups, names, dtypes)
        for name, dt in zip(grnames, dtype)
    }
    ic0(history)
    history["trackids:TrackId"] = track_ids
    history["trackids:TrackId_current_host"] = host_track_id_today
    history = pd.DataFrame(history).reset_index()
    ic0(np.sort(history.columns))
    ic0(history.shape)
    # testing
    # jtr = (history['trackids:TrackId'] == 31158)
    # jtr_cols = ['trackids:TrackId', 'max_Mstar:isnap',
    #             'max_Mstar:time', 'max_Mstar:Mstar']
    # ic0(jtr.sum())

    # save everything to an hdf5 file
    path_history = os.path.join(sim.data_path, "history")
    hdf = os.path.join(path_history, "history.h5")
    if args.test:
        hdf = hdf.replace(".h5", ".test.h5")
    if os.path.isfile(hdf):
        os.system(f"cp -p {hdf} {hdf}.bak")
        os.system(f"rm {hdf}")

    ## now calculate things!
    snaps = np.sort(sim.snapshots)[::-1]
    # snaps = np.concatenate([snaps[:360:40], snaps[-5:]])
    ic(snaps, snaps.size)

    # a flag to update pericenters
    past_peri = np.zeros(track_ids.size, dtype=bool)

    z, t = np.zeros((2, snaps.size))
    failed = []
    for i, snap in enumerate(tqdm(snaps)):
        ics["history"](i, snap)
        subs_i = reader.LoadSubhalos(snap)
        # snapshot 281 is missing
        if subs_i.size == 0:
            ics["history"]("no data")
            continue
        subs_i = Subhalos(
            subs_i,
            sim,
            snap,
            logMmin=None,
            logM200Mean_min=None,
            logMstar_min=None,
            load_any=False,
            verbose_when_loading=False,
        )
        ic0(type(subs_i), subs_i.shape, subs_i["Mstar"].min(), subs_i["Mstar"].max())
        z[i] = sim.redshift(snap)
        t[i] = sim.cosmology.lookback_time(z[i]).to("Gyr").value

        # first match subhaloes now and then
        ics["history"](history.shape)
        this = history.merge(
            subs_i.catalog,
            left_on="trackids:TrackId",
            right_on="TrackId",
            how="left",
            suffixes=("_test", "_snap"),
        )
        ics["history"](this.shape)
        this = Subhalos(
            this,
            subs_i.sim,
            subs_i.isnap,
            load_any=False,
            logMmin=None,
            logM200Mean_min=None,
            logMstar_min=None,
            exclude_non_FoF=False,
            verbose_when_loading=False,
        )
        if i == 0:
            ic0(np.sort(this.columns))
        ics["history"](this.shape, subs_i.shape, history.shape)

        # this raises a MemoryError
        # this.distance2host('Comoving')

        ## sat
        # this one gets updated all the time
        still_sat = np.array(this["Rank"] > 0)
        ics["sat"](still_sat.sum())
        # history.loc[still_sat, cols['sat'][:3]] = [snap, t[i], z[i]]
        # history = assign_masses(history, this, still_sat, 'sat')
        history = update_history(
            history, this, cols, snap, t[i], z[i], still_sat, "sat"
        )

        ## cent
        # this one is updated only once per subhalo
        if i > 0:
            new_cent = (this["cent:isnap"] == -1) & (this["Rank"] == 0)
            ics["cent"](new_cent.sum())
            history.loc[new_cent, cols["cent"][:3]] = [snap, t[i], z[i]]
            history = assign_masses(history, this, new_cent, "cent")
            all_cent_history = history["cent:isnap"] > -1
            ic_cols = ["trackids:TrackId"] + [
                f"cent:{x}" for x in ("isnap", "z", "time", "Mbound", "Mstar")
            ]
            # ics['cent'](history.loc[all_cent_history, ic_cols])
        # continue

        history = max_mass(history, this, groups, cols, snap, t[i], z[i])
        # ics['masses'](this.catalog.loc[jtr, ['TrackId','MboundType4']])
        # ics['masses'](history.loc[jtr, jtr_cols])
        test = history["trackids:TrackId"] == 25270
        ics["masses"](subs_i["TrackId", "Mstar"][subs_i["TrackId"] == 25270])

        ## birth
        # this is updated all the time while the subhalo exists
        # doing this every time so we can record the mass at birth
        # and so that satellites that already existed in the first snapshot
        # do get assigned a birth time
        exist = np.isin(history["trackids:TrackId"], subs_i["TrackId"])
        ics["birth"](exist.sum(), (1 - exist).sum())
        # history.loc[exist, cols['birth'][:3]] = [snap, t[i], z[i]]
        # history = assign_masses(history, this, exist, 'birth')
        history = update_history(history, this, cols, snap, t[i], z[i], exist, "birth")

        ## infall
        # first map HostHaloId to TrackId
        # seems like I need to ask for HostHaloId > -1 to avoid
        # multiple matches but what if a present-day satellite
        # was last a central with HostHaloId = -1?
        this_cent_mask = (subs_i["Rank"] == 0) & (subs_i["HostHaloId"] > -1)
        this_cent = subs_i.catalog.loc[this_cent_mask, ["TrackId", "HostHaloId"]]
        ics["infall"](this_cent_mask.shape, this_cent_mask.sum())
        ics["infall"](this.shape, this_cent.shape)
        this.merge(
            this_cent,
            on="HostHaloId",
            how="left",
            suffixes=("_sat", "_cent"),
            in_place=True,
        )
        ics["infall"](this.shape)
        # first update the masses for all subhalos for which we have
        # not registered infall already
        jinfall_last = (this["last_infall:isnap"] == -1).values
        ics["infall"](jinfall_last.shape, jinfall_last.sum())
        history = assign_masses(history, this, jinfall_last, "last_infall")
        # if they are no longer in the same host, then register
        # the other columns. Once we've done this their masses
        # will never be updated again
        host_track = history["trackids:TrackId_current_host"]
        # this is not happening anymore, keeping here just in case
        try:
            jinfall_last = jinfall_last & (this["TrackId_cent"] != host_track)
        except ValueError as e:
            ics["infall"](
                jinfall_last.shape,
                jinfall_last.sum(),
                this["TrackId_cent"].shape,
                host_track.shape,
            )
            print(f"*** ValueError({e}) ***")
            failed.append(snap)
            if len(failed) == 3:
                print(f"First failed: {failed}. Exiting.")
                return
            raise ValueError(e)
            continue
        #
        ics["infall"](jinfall_last.sum())
        history.loc[jinfall_last, cols["last_infall"][:3]] = [
            snap + 1,
            t[i - 1],
            z[i - 1],
        ]
        ics["history"](jinfall_last.values)
        ics["history"](this["Depth"].values, this.shape)
        ics["history"](this["Depth"].values[jinfall_last.values])
        history.loc[jinfall_last, cols["last_infall"][-1]] = this["Depth"].values[
            jinfall_last.values
        ]

        # assign these same values to first_infall
        # history = assign_masses(history, this, jinfall_last, 'first_infall')
        # history.loc[jinfall_last, cols['first_infall'][:3]] \
        #     = [snap+1, t[i-1], z[i-1]]
        history = update_history(
            history,
            this,
            cols,
            snap + 1,
            t[i - 1],
            z[i - 1],
            jinfall_last,
            "first_infall",
        )
        # here we record the first time each subhalo fell into its
        # current host, i.e., the same as above but the info can be
        # updated once set
        left_host_and_back = (history["last_infall:isnap"] != -1) & (
            this["TrackId_cent"] == host_track
        )
        ics["infall"](left_host_and_back.sum())
        # history.loc[left_host_and_back, cols['first_infall'][:3]] \
        #     = [snap, t[i], z[i]]
        # history = assign_masses(
        #     history, this, left_host_and_back, 'first_infall')
        history = update_history(
            history, this, cols, snap, t[i], z[i], left_host_and_back, "first_infall"
        )
        # TrackId_previous_host refers to the host prior to the first
        # time the subhalo entered its current host (does it make a
        # difference?), i.e. those for which we've already registered
        # an infall in the *immediate* previous iteration
        # this will be updated each time the subhalo leaves
        jinfall_first = (history["first_infall:isnap"] == snap + 1) & (
            this["TrackId_cent"] != host_track
        )
        ics["infall"](jinfall_first.sum())
        history.loc[jinfall_first, "trackids:TrackId_previous_host"] = this[
            "TrackId_cent"
        ][jinfall_first]

        # pericenter
        # past_

        if args.store and ((i % 10 == 0) or (snap < 10)):
            store_h5(hdf, groups, names, history)
        if args.test and i >= 3:
            break

    if args.test or args.debug:
        ics["masses"]()
        mcols = cols["max_Mstar"]
        mcols.insert(0, "trackids:TrackId")
        # test_track = (history['TrackId'] ==
        # ics['masses'](history.)
        ics["masses"](history.iloc[:20][mcols])
        ics["masses"](np.histogram(history["max_Mstar:time"], np.arange(0, 15, 2))[0])
        ics["history"]()
        for key, val in cols.items():
            if "Depth" in val[-1]:
                ics["history"](key, val[-1], history.iloc[:20][val[-1]])

    print(f"Failed: {failed}, i.e., {len(failed)} times")
    if args.store or args.test:
        store_h5(hdf, groups, names, history)
        print(f"Finished! Stored everything to {hdf}")

    return


def assign_masses(history, this, mask, group):
    for m in ("Mbound", "Mgas", "Mdm", "Mstar"):
        history.loc[mask, f"{group}:{m}"] = this[m][mask]
    return history


def max_mass(history, this, events, cols, snap, ti, zi):
    ics["masses"]()
    ics["masses"](np.sort(events))
    for event in events:
        if event[:4] == "max_":
            m = event[4:].split("/")
            histdata = history[f"{event}:{m[0]}"]
            thisdata = this[m[0]]
            if len(m) == 2:
                histdata = histdata / history[f"{event}:{m[1]}"]
                thisdata = thisdata / this[m[1]]
            m = "/".join(m)
            # greater or eaqual than because I'm going backwards
            # in time (i.e., I want to record the earliest snapshot
            # where the maximum mass is obtained)
            gtr = thisdata >= histdata
            # history.loc[gtr, cols[event][:3]] = [snap, ti, zi]
            # history = assign_masses(history, this, gtr, event)
            history = update_history(history, this, cols, snap, ti, zi, gtr, event)
            if m in ("Mstar", "Mbound"):
                mcols = [i for i in cols[event]]
                mcols.insert(0, "trackids:TrackId")
                ics["masses"](snap, history.loc[gtr, mcols][:10], gtr.sum())
    return history


def store_h5(hdf, groups, names, data):
    with h5py.File(hdf, "w") as out:
        for group, grnames in zip(groups, names):
            gr = out.create_group(group.replace("/", "-over-"))
            for name in grnames:
                col = f"{group}:{name}"
                if group == "max_Mstar":
                    ics["masses"](col, data[col].to_numpy())
                gr.create_dataset(
                    name, data=data[col].to_numpy(), dtype=data[col].dtype
                )
    ic0(f"Saved to {hdf}")
    return


# def store_npy(path, groups, names, history):
#     np.save(output['trackids'],
#             [tracks.track_ids, TrackId_current_host,
#              TrackId_previous_host])
#     np.save(output['idx'], [idx_infall, idx_sat, idx_cent])
#     np.save(output['t'], [t_infall, t_sat, t_cent])
#     np.save(output['z'], [z_infall, z_sat, z_cent])
#     np.save(output['Mbound'], [Mbound_infall, Mbound_sat, Mbound_cent])
#     np.save(output['Mstar'], [Mstar_infall, Mstar_sat, Mstar_cent])
#     np.save(output['Mdm'], [Mdm_infall, Mdm_sat, Mdm_cent])
#     return


def update_history(history, subs, cols, snap, t, z, mask, label):
    # times
    history.loc[mask, cols[label][:3]] = [snap, t, z]
    # depth
    history.loc[mask, cols[label][-1]] = subs["Depth"][mask]
    # masses
    history = assign_masses(history, subs, mask, label)
    return history


if __name__ == "__main__":
    main()
