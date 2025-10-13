import argparse
import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import einops as ein
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1 import ImageGrid

from utils.metrics import expert_overlap_max, expert_overlap_min

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
NUM_EXP = WIDTH * HEIGHT

# load stuff
magic_idxs = np.load(data_dir / "magic_idxs.npy")
bitmaps = np.load(data_dir / "bitmaps.npy")
labels = np.load(data_dir / "labels.npy")


dlen = len(bitmaps)
magic_bitmaps = bitmaps[np.arange(dlen), magic_idxs]
active_experts = ein.rearrange(magic_bitmaps, "b l e -> b (l e)")
classwise_overlaps_min = [
    expert_overlap_min(active_experts[labels == i]) for i in range(NUM_CLASSES)
]
classwise_overlaps_max = [
    expert_overlap_max(active_experts[labels == i]) for i in range(NUM_CLASSES)
]


fig = plt.figure(figsize=(6.5, 4))
fig.suptitle("Max. overlap")
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(2, 5),
    share_all=True,
    axes_pad=0.4,
    aspect=False,
)
for lab, (ov, ax) in enumerate(zip(classwise_overlaps_max, grid)):
    counts, bins = np.histogram(ov, density=True)
    ax.stairs(counts, bins)
    ax.set_xlim(0, 1)
    ax.set_title(f"Class: {lab}")
    ax.set_xlabel("Overlap value")
    ax.set_ylabel("Density")
    ax.autoscale()
plt.savefig(plot_dir / "overlaps_max.pdf")


fig = plt.figure(figsize=(6.5, 4))
fig.suptitle("Min. overlap")
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(2, 5),
    share_all=True,
    axes_pad=0.4,
    aspect=False,
)
for lab, (ov, ax) in enumerate(zip(classwise_overlaps_min, grid)):
    counts, bins = np.histogram(ov, density=True)
    ax.stairs(counts, bins)
    ax.set_xlim(0, 1)
    ax.set_title(f"Class: {lab}")
    ax.set_xlabel("Overlap value")
    ax.set_ylabel("Density")
    ax.autoscale()
plt.savefig(plot_dir / "overlaps_min.pdf")
