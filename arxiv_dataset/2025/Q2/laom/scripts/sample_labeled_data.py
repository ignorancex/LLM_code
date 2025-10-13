# We used this script to sample labelled trajectories from the full dataset.
# Although full dataset also have ground-truth actions (debug only),
# only these sampled datasets were considered labelled.
import os
import random
from dataclasses import dataclass

import h5py
import pyrallis

from src.utils import set_seed


@dataclass
class Args:
    data_path: str = "data.hdf5"
    save_path: str = "labeled_data.hdf5"
    chunk_size: int = 1000
    num_trajectories: int = 32
    seed: int = 42


def copy_attrs(new_df, df):
    for key in df.attrs:
        new_df.attrs[key] = df.attrs[key]


@pyrallis.wrap()
def main(args: Args):
    set_seed(seed=args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    assert args.chunk_size >= 0
    new_df = h5py.File(args.save_path, "w")

    with h5py.File(args.data_path, "r") as df:
        copy_attrs(new_df, df)

        total_trajectories = len(df.keys())
        # sampling without replacement
        sampled_trajectories = list(range(total_trajectories))
        random.shuffle(sampled_trajectories)
        sampled_trajectories = sampled_trajectories[: args.num_trajectories]

        for i, traj_idx in enumerate(sampled_trajectories):
            traj_idx = str(traj_idx)
            start_idx = random.randint(0, 1000 - args.chunk_size)

            group = new_df.create_group(str(i))
            # copy group attrs & save source traj
            copy_attrs(group, df[traj_idx])
            group.attrs["original_traj_idx"] = traj_idx

            # copy data
            group.create_dataset(
                "obs",
                shape=(args.chunk_size, *df[traj_idx]["obs"].shape[1:]),
                data=df[traj_idx]["obs"][start_idx : start_idx + args.chunk_size],
            )
            group.create_dataset(
                "states",
                shape=(args.chunk_size, *df[traj_idx]["states"].shape[1:]),
                data=df[traj_idx]["states"][start_idx : start_idx + args.chunk_size],
            )
            group.create_dataset(
                "actions",
                shape=(args.chunk_size, *df[traj_idx]["actions"].shape[1:]),
                data=df[traj_idx]["actions"][start_idx : start_idx + args.chunk_size],
            )

    new_df.close()


if __name__ == "__main__":
    main()
