import numpy as np

# Base directory where all runs are stored
runs_location = "../data"

def get_data(filename: str, run: str) -> tuple[np.ndarray, np.ndarray]:
   simulation_directory = f"{runs_location}/{run}"
   redshift = np.load(f"{simulation_directory}/redshift.npy")
   data = np.load(f"{simulation_directory}/{filename}.npy")[:-1]
   return redshift, data


import matplotlib.pyplot as plt

# Load data

get_data_group = lambda region: get_data(
   f"gas_history_entropy_R{region}_Thot_Oref",
   "VR2915_+1res_ref"
)


redshift, core = get_data_group("core")
_, shell = get_data_group("shell")
_, field = get_data_group("field")

# Normalize: divide by k500 at z=0
k500 = np.load("../data/VR2915_+1res_ref/k500.npy")[-1]  # In keV cm^2

core /= k500
shell /= k500
field /= k500

# Cut out very high-z data
z_mask = redshift < 4.0
redshift = redshift[z_mask]
core = core[z_mask]
shell = shell[z_mask]
field = field[z_mask]

plt.figure()
plt.loglog(redshift + 1, core, lw=3, label="Core")
plt.loglog(redshift + 1, shell, lw=3, label="Plateau")
plt.loglog(redshift + 1, field, lw=3, label="Field")
plt.xlabel("Redshift + 1")
plt.legend()
plt.ylabel(r"$K/K_{500}(z=0)$")
plt.title("Lagrangian entropy evolution")

plt.savefig("lagrangian_histories.png")