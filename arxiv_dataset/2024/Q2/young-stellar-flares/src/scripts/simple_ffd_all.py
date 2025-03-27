import os
import numpy as np
import matplotlib.pyplot as plt

from paths import (
    data as data_dir,
    figures as figures_dir
)

pwd1 = os.path.join(data_dir, 'mcmc_fits/samples')
pwd2 = os.path.join(data_dir, 'mcmc_fits/fitted_data')

sample_files = np.sort([os.path.join(pwd1, i) for i in os.listdir(pwd1)])
data_files = np.sort([os.path.join(pwd2, i) for i in os.listdir(pwd2)])
sample_sizes = np.load(os.path.join(data_dir,
                                    'mcmc_fits/mcmc_sample_sizes_v6.npy'),
                       allow_pickle=True).item()

## Setup the figure ##
fig, axes = plt.subplots(ncols=5, nrows=7, figsize=(20,15),
                         sharex=True, sharey=True)
axes = axes.reshape(-1)

## Loop through the files ##
for i, fn in enumerate(sample_files):
    # Get the age range from the data file
    name = fn.split('/')[-1]
    fn_ages = name.split('_')[2].split('-')
    fn_ages = [float(fn_ages[0][1:]), float(fn_ages[1])]

    # Get the teff range from the data file
    fn_teff = name.split('_')[3].split('-')
    fn_teff = [float(fn_teff[0][1:]), float(fn_teff[1])]

    # Nflares & Nstars in sample
    nflares = sample_sizes[name[13:]][0]
    nstars = sample_sizes[name[13:]][1]

    # Load the fitted data
    fitted_data = np.load(data_files[i], allow_pickle=True)
    all_data = fitted_data[0]
    fitted_data = fitted_data[1]

    # Load the MCMC samples
    flat_samples = np.load(fn)

    # Plot some best-fit lines, drawn randomly from the samples
    inds = np.random.randint(len(flat_samples), size=100)
    if len(fitted_data[:,0]) > 3:
        for ind in inds:
            sample = flat_samples[ind]
            axes[i].plot(fitted_data[:,0], np.dot(np.vander(fitted_data[:,0], 2),
                         sample[:2]), color='#e36414', alpha=0.05)

    # Plot all of the data
    axes[i].errorbar(all_data[:,0],
                     all_data[:,1],
                     yerr=all_data[:,2], color='#c0d6df',
                     marker='o', linestyle='', zorder=200)

    # Plot the data that was fitted
    axes[i].errorbar(fitted_data[:,0],
                     fitted_data[:,1],
                     yerr=fitted_data[:,2], color='#0f1108',
                     marker='o', linestyle='', zorder=200)

    axes[i].text(s=r'$N_{flares}$ = ' + str(nflares),
                 x=35, y=-1.5, fontsize=12, ha='right')
    axes[i].text(s=r'$N_{stars}$ = ' + str(nstars),
                 x=35, y=-1.9, fontsize=12, ha='right')

    # Make some titles
    if i < 5:
        axes[i].set_title('{0}-{1} K'.format(int(fn_teff[0]), int(fn_teff[1])),
                          fontsize=14, fontweight='bold')

    # Set some ylabels
    if ((i == 4) | (i == 9) | (i == 14) | (i == 19) |
        (i == 24) | (i == 29) | (i == 34)):
        axes[i].yaxis.set_label_position('right')
        axes[i].set_ylabel('{0}-{1} Myr'.format(int(fn_ages[0]), int(fn_ages[1])),
                           fontsize=14, fontweight='bold')

plt.xlim(28,35.5)
plt.ylim(-4, -1)
plt.xticks([29, 32, 35])
axes[-3].set_xlabel(r'log$_{10}$(Flare Energy [erg])', fontsize=24)
axes[15].set_ylabel(r'log$_{10}$(Flare Rate [day$^{-1}$])', fontsize=24)

for ax in axes:
    ax.set_rasterized(True)

plt.savefig(os.path.join(figures_dir, 'simple_ffd_all.pdf'),
            dpi=250, bbox_inches='tight')
