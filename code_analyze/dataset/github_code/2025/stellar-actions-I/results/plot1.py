## first plot of the paper uses data not available here so code just for reference
#fig 1. snapshot of simulation plot

import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
os.chdir('/g/data/jh2/ax8338/action/action_function/')
import functions as f


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

i=564
fileName='/scratch/jh2/cz4203/final_simulation/snapshots/snapdir_{:03d}/'.format(i, i)
C, ID, M, V, A, Pot = f.read_stars(fileName)
C_gas, ID_gas, V_gas, M_gas, Pot_gas, A_gas = f.read_gas(fileName)
mid_mask = (M>20)&(M<500)
C_mid,ID_mid, M_mid, V_mid, A_mid, Pot_mid = C[mid_mask], ID[mid_mask], M[mid_mask], V[mid_mask], A[mid_mask], Pot[mid_mask]

ID_star_list = np.loadtxt('/g/data/jh2/ax8338/action/early_stars_removed.txt')
random_10k = random.sample([*ID_star_list],int(13200))

# Create a dictionary for quick ID lookup
id_to_index = {id_: idx for idx, id_ in enumerate(ID_mid)}

# Filter and maintain order in `random_10k`
indices = [id_to_index[id_] for id_ in random_10k if id_ in id_to_index]

# Extract the corresponding positions and ages
C_selected = C_mid[indices]
A_selected = A_mid[indices]

# Verify
print(f"Number of stars selected: {C_selected.shape[0]}")

A_selected= A_selected[C_selected[:,0]<18000]
C_selected=C_selected[C_selected[:,0]<18000]


# Assuming C_gas, M_gas, and C_selected, A_selected are defined
x_range = (-18, 18)
y_range = (-18, 18)
bins = 400

# Create histogram
H, x_edges, y_edges = np.histogram2d(
    C_gas[:, 0] / 1000,
    C_gas[:, 1] / 1000,
    bins=bins,
    range=[x_range, y_range],
    weights=M_gas,
)

# Calculate bin area
dx = (x_range[1] - x_range[0]) / bins
dy = (y_range[1] - y_range[0]) / bins
bin_area = dx * dy

# Compute surface density
surface_density = H / bin_area  # Mass per unit area

# Set up the figure and GridSpec for square plot and colour bar
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.1)

# Main plot
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(
    surface_density.T,
    origin="lower",
    extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
    cmap="gray_r",
    aspect="equal",
    norm=mcolors.LogNorm(),
    alpha=1,
)

# Scatter plot for stars
scatter = ax.scatter(
    C_selected[:, 0] / 1000,  # x positions
    C_selected[:, 1] / 1000,  # y positions
    c=A_selected,             # colour by age
    cmap="plasma",            # or any other colormap
    s=4,                      # marker size
    edgecolor="none",
    alpha=1,
)

# Colour bar for scatter plot
cbar = fig.colorbar(scatter, cax=fig.add_subplot(gs[0, 1]))
cbar.set_label("Age of Stars [Myr]")

# Labels and equal aspect ratio
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_aspect('equal')  # Ensure the axes are square
# Save or show the plot
plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/snapshot1_final.pdf')
plt.show()
