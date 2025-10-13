import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# load files
parser = argparse.ArgumentParser(description="Plot expert bitmap")
parser.add_argument(
    "--config",
    type=str,
    default="default_run",
    help="Path to the config file.",
)
args = parser.parse_args()
data_dir = Path("../data") / args.config
plot_dir = Path(".") / args.config
cfg_dir = Path("../configs/")
if args.config:
    with open(cfg_dir / f"{args.config}.yaml") as f:
        config = yaml.safe_load(f)

expert_frs = np.load(data_dir / "experts_frs.npy")
magic_idxs = np.load(data_dir / "magic_idxs.npy")
labels = np.load(data_dir / "labels.npy")

expert_frs = expert_frs[np.arange(len(expert_frs)), magic_idxs].reshape(len(expert_frs), -1)[:, :5]
print(expert_frs.shape)

pca = PCA(2)
print(expert_frs[1])
X = pca.fit_transform(expert_frs+ 1e-8)
fig = plt.figure(figsize=(6.5, 6))
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', edgecolor='k', s=40)
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.title("TSNE")
plt.colorbar(scatter, label='Label')
plt.grid(True)
plt.show()