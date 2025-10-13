import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from itertools import product
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
cfg_dir = Path("../configs/")

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
shape = active_experts.shape[1:]
act_per_expert:np.ndarray = active_experts.sum(0) / len(active_experts)
print(act_per_expert)
act_per_expert = np.append(act_per_expert, np.mean(act_per_expert))
fig, ax = plt.subplots(figsize=(6.5, 6))
nums = np.arange(len(act_per_expert))
names = [f"l {i+1}, e {j+1}" for (i,j) in product(range(shape[0]), range(shape[1]))] + ["global average"]
ax.bar(nums, act_per_expert, label=nums)
ax.set_xticks( nums + 0.5)
ax.set_xticklabels(names) 
plt.xticks(rotation=45, ha="right")
ax.set_xlabel("Num. active experts")
ax.set_ylabel("Counts")
plt.savefig(plot_dir / "act_per_experts.pdf")
