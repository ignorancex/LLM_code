import h5py
import numpy as np


def load_hdf5_to_dict(filepath):
    data_dict = {}

    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data = f[key][()]

            # Automatically convert to numpy array if needed
            if isinstance(data, np.ndarray):
                data_dict[key] = data
            else:
                data_dict[key] = np.array(data)

    return data_dict


group_data = load_hdf5_to_dict("../data/fig2_maps/group_+8res_maps.hdf5")
cluster_data = load_hdf5_to_dict("../data/fig2_maps/cluster_+1res_maps.hdf5")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

phase_idx = -1
data = group_data["temperature_mass_hightemp"][phase_idx] / group_data["masses_hightemp"][phase_idx] / 1e10

plt.axis("off")
plt.imshow(data, norm=LogNorm(), cmap='twilight')
plt.colorbar(label="Temperature [Kelvin]")
plt.tight_layout()
plt.savefig("map_group.png")
plt.close()

data = cluster_data["temperature_mass_hightemp"][phase_idx] / cluster_data["masses_hightemp"][phase_idx] / 1e10

plt.axis("off")
plt.imshow(data, norm=LogNorm(), cmap='twilight')
plt.colorbar(label="Temperature [Kelvin]")
plt.tight_layout()
plt.savefig("map_cluster.png")