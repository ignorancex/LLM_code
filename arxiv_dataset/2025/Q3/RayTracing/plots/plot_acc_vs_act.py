import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.style.use("myplots.mplstyle")

# load files
parser = argparse.ArgumentParser(
    description="Plot accuracy vs output node's firing rate at activation"
)
parser.add_argument(
    "--config",
    type=str,
    help="Path to the config file.",
)
parser.add_argument(
    "--num_percs",
    type=int,
    default=20,
    help="How many percentiles for the curve aggregation.",
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
preds = np.load(data_dir / "all_preds.npy")
acts_seqs = np.load(data_dir / "acts_seqs.npy")
firing_rates = np.load(data_dir / "experts_frs.npy")
magic_idxs = np.load(data_dir / "magic_idxs.npy")
bitmaps = np.load(data_dir / "bitmaps.npy")
labels = np.load(data_dir / "labels.npy")


#
len_test = len(preds)
ps = np.linspace(0, 100, args.num_percs)
fig, ax = plt.subplots(figsize=(6.5, 6))
acts = acts_seqs[np.arange(len_test), magic_idxs]
predictions = preds[np.arange(len_test), magic_idxs].argmax(-1)
is_correct = predictions == labels
percs = np.percentile(acts, ps)
idxs = np.digitize(acts, percs)
nums = np.array([(idxs == i).sum() for i in range(1, len(ps) + 1)])
mean_acts = np.array([acts[idxs == i].mean() for i in range(1, len(ps) + 1)])
mean_accs = np.array([is_correct[idxs == i].mean() for i in range(1, len(ps) + 1)])
se = np.sqrt(mean_accs * (1 - mean_accs) / nums)
std_accs = np.array([is_correct[idxs == i].std() for i in range(1, len(ps) + 1)])
ci_95 = 1.96 * se
ax.plot(mean_acts, mean_accs, "o-", label="Mean")
ax.fill_between(
    mean_acts, mean_accs - ci_95, mean_accs + ci_95, alpha=0.5, label="$95\%$ C.I."
)
ax.set_xlabel("Output firing rate")
ax.set_ylabel("Prediction accuracy")
ax.legend()
plt.savefig(plot_dir / "acc_vs_act.pdf")
plt.close()
