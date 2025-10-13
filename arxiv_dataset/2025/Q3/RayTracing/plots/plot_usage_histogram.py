import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.style.use("myplots.mplstyle")


# load files
parser = argparse.ArgumentParser(description="Plot expert bitmap")
parser.add_argument(
    "--config",
    type=str,
    help="Path to the config file.",
)
args = parser.parse_args()
data_dir = Path("../data") / args.config
plot_dir = Path(".") / args.config
plot_dir.mkdir(parents=True, exist_ok=True)
cfg_dir = Path("../configs/raytracing")

if args.config:
    with open(cfg_dir / f"{args.config}.yaml") as f:
        config = yaml.safe_load(f)

NUM_CLASSES = 10
expert_grid = [int(i) for i in config["n_exps"].split(" ")]
WIDTH = expert_grid[0]
HEIGHT = len(expert_grid)

# load stuff
magic_idxs = np.load(data_dir / "magic_idxs.npy")
bitmaps = np.load(data_dir / "bitmaps.npy")
labels = np.load(data_dir / "labels.npy")

dlen = len(bitmaps)
active_experts = bitmaps[np.arange(dlen), magic_idxs]
num_active_experts = np.sum(active_experts, axis=(-1, -2))
counts = np.bincount(num_active_experts, minlength=sum(expert_grid))
counts = np.roll(counts, -1)

fig, ax = plt.subplots(figsize=(6.5, 6))
nums = np.arange(1, sum(expert_grid) + 1)
ax.bar(nums, counts, label=nums)
ax.set_xticks(nums)
ax.set_xlabel("Num. active experts")
ax.set_ylabel("Counts")
plt.savefig(plot_dir / "usage_histogram.pdf")
