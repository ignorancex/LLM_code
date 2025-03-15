##getting pre-calculated data from the data directory to get the plot 11 of the paper- dependence of action change on birth radius
### get the radial_bin zipfile from the following website, unzip it and change the file paths here accordingly ('stellar-actions-I/data/plot_data/radial_bin/' in following code):
# https://www.mso.anu.edu.au/~arunima/stellar-actions-I-data/radial_bin.zip

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
import glob
from analysis.utils import adding_histograms_from_chunks

 # Orbital period for each radial bin - can read in from the file created by analysis/identify_radial_bins_calculate_change.py
radial_period = [
    np.float64(92.87450079416625),
    np.float64(105.30116214696899),
    np.float64(131.08304111035918),
    np.float64(159.97576600898213),
    np.float64(182.79652790066905),
    np.float64(209.7661172745036),
    np.float64(243.0990700167341),
    np.float64(276.00865032521625),
    np.float64(297.5305353289982),
    np.float64(327.2493205448509),
    np.float64(352.6837999195652),
    np.float64(386.86877116869726),
    np.float64(419.6876114675658),
    np.float64(441.0578947318629),
    np.float64(463.8257913884764),
    np.float64(472.1416765563095),
    np.float64(498.51928114735983)
]

# Set up colormap
colormap = cm.get_cmap('viridis', len(range(2, 15, 2)))
colors = colormap(np.linspace(0, 1, len(range(2, 15, 2))))
plt.close('all')

# Function to load the data once for a given component (R or z) and l_R: radial bin number
def load_data(component, l_R):
    if component == "R":
        glob_word = f'filtabs_J_R_{l_R}kpc*'
    elif component == "z":
        glob_word = f'*vz_Jz_{l_R}kpc*'
    
    # HDF5 file list (change path here)
    hdf5_file_list = glob.glob(f'stellar-actions-I/data/plot_data/radial_bin/{glob_word}.h5')
    
    # adding_histograms_from_chunks function 
    adding_trial, age_bins, delta_t_bins, bin_centres = adding_histograms_from_chunks(
        hdf5_file_list, bin_size=5999
    )
    
    return adding_trial, age_bins, delta_t_bins, bin_centres

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')

# Loop through all l_R values (from 2 to 14 in steps of 2) and plot for both R and z
for idx, l_R in enumerate(np.arange(2,15,2)):
    # Load data for R and z (only once for each l_R)
    data_R = load_data("R", l_R)
    data_z = load_data("z", l_R)
    
    # Extract values for plotting
    adding_trial_R, age_bins_R, delta_t_bins_R, bin_centres_R = data_R
    adding_trial_z, age_bins_z, delta_t_bins_z, bin_centres_z = data_z

    # Calculate medians for J_R
    histogram_delta_t_R = adding_trial_R.sum(axis=0)
    medians_ageless_R = np.zeros(len(delta_t_bins_R))

    for delta_t_bin in tqdm(range(len(delta_t_bins_R))):
        cdf_R = np.cumsum(histogram_delta_t_R[delta_t_bin, :]) / np.sum(histogram_delta_t_R[delta_t_bin, :])
        medians_ageless_R[delta_t_bin] = np.interp(0.5, cdf_R, bin_centres_R)
    
    medians_ageless_R = medians_ageless_R[1:]
    delta_t_bins_R = delta_t_bins_R[1:]

    # Calculate medians for J_z
    histogram_delta_t_z = adding_trial_z.sum(axis=0)
    medians_ageless_z = np.zeros(len(delta_t_bins_z))

    for delta_t_bin in tqdm(range(len(delta_t_bins_z))):
        cdf_z = np.cumsum(histogram_delta_t_z[delta_t_bin, :]) / np.sum(histogram_delta_t_z[delta_t_bin, :])
        medians_ageless_z[delta_t_bin] = np.interp(0.5, cdf_z, bin_centres_z)
    
    medians_ageless_z = medians_ageless_z[1:]
    delta_t_bins_z = delta_t_bins_z[1:]

    # Plot for absolute time (both R and z)
    axes[0, 0].plot(delta_t_bins_R, medians_ageless_R**2, '-', color=colors[idx], label=f"{l_R}-{l_R+1} kpc")
    axes[1, 0].plot(delta_t_bins_z, medians_ageless_z**2, '-', color=colors[idx], label=f"{l_R}-{l_R+1} kpc")
    
    # Plot for orbital time (both R and z) - use radial_period[l_R-1] 
    axes[0, 1].plot(delta_t_bins_R / radial_period[l_R - 1], medians_ageless_R**2, '-', color=colors[idx])
    axes[1, 1].plot(delta_t_bins_z / radial_period[l_R - 1], medians_ageless_z**2, '-', color=colors[idx])

# labels
axes[1, 0].set_xlabel(r"$\Delta t$ (Myr)")
axes[1, 1].set_xlabel(r"$\Delta t/ \tau_{orb}$")

axes[0, 0].set_ylabel(r"$\langle \delta J_{R}^2 \rangle$")
axes[1, 0].set_ylabel(r"$\langle \delta J_{z}^2 \rangle$")

# Set the x-limits
axes[0, 0].set_xlim(0, 200)  # Absolute time
axes[1, 0].set_xlim(0, 200)  # Absolute time
axes[1, 1].set_xlim(0, 0.6)  # Orbital time
axes[0, 1].set_xlim(0, 0.6)  # Orbital time

axes[0, 0].set_ylim(0, 0.6)  # Absolute time
axes[1, 0].set_ylim(0, 0.6)  # Absolute time
axes[1, 1].set_ylim(0, 0.6)  # Orbital time
axes[0, 1].set_ylim(0, 0.6) 

# Display legend only in the first plot
axes[0, 0].legend(ncol=2)
axes[0,1].legend()
axes[1,1].legend()

# Add grid to all subplots
for ax in axes.flatten():
    ax.grid(True)

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# Save the figure if needed
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/radial_bins.pdf', dpi=300)
