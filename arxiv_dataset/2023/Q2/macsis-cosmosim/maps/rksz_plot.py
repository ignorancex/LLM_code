import os
import sys
import unyt
import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle

# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
        )
    )
)

from read import MacsisDataset
from register import Macsis

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

Tcmb0 = 2.7255
projections = ['x', 'y', 'z', 'faceon', 'edgeon']
display_maps = dict()
macsis = Macsis()

print("Collecting data from file...")
with h5py.File(f'{macsis.output_dir}/rksz_gas.hdf5', 'r') as f:
    for projection in projections:
        for i, halo in enumerate(tqdm(f.keys(), desc=f"Merging map_{projection}")):
            dataset = f[f"{halo}/map_{projection}"][:]
            if i == 0:
                display_maps[projection] = dataset
            else:
                display_maps[projection] += dataset

# Get maximum limit and apply to all colormaps
max_list = []
for p in display_maps:
    max_list.append(np.abs(display_maps[p]).max())
vlim = max(max_list)
print(vlim)

print("Composing plot figure...")
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True,
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

for ax, projection, smoothed_map in zip(axes.flat, projections, display_maps.values()):
    im = ax.imshow(
        smoothed_map,
        norm=SymLogNorm(linthresh=1e-1, vmin=-vlim, vmax=vlim),
        cmap="PRGn",
        origin="lower",
        extent=(-2, 2, -2, 2),
    )
    ax.set_axis_off()
    ax.set_title(f"Projection {projection}")
    r500_circle = Circle((0, 0), 1, fill=False, linewidth=1, linestyle='-', color='k')
    ax.add_artist(r500_circle)
    ax.text(0, 1, r'$R_{500, crit}$',
            horizontalalignment='center',
            verticalalignment='bottom',
            # transform=ax.transAxes,
            color='k')

plt.title('All clusters')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, label=r'$\sum y_{ksz}$')
fig.tight_layout()
plt.savefig(f'{macsis.output_dir}/rksz_gas.png', dpi=350)
plt.show()
plt.close()
