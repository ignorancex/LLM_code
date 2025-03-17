import os
import sys
import matplotlib
import numpy as np
from astropy import units
from astropy.table import Table
import matplotlib.pyplot as plt

from plotting_utils import *
from data_utils import *

from paths import (
    figures as figures_dir,
    data as data_dir,
)

set_rcparams()
teff_cbar = teff_colorbar().reversed()

temp_ranges = np.array([[2300, 3400], [3400, 3850],
                        [3850, 4440], [4440, 5270], [5270, 5930]])
age_ranges  = np.array([[4, 10], [10, 20], [20,40], [40, 50],
                        [70, 80], [120, 150], [150, 300]])

# Load in the files
path = os.path.join(data_dir, 'mcmc_fits/best_fits')
fits = np.sort([os.path.join(path, i) for i in os.listdir(path)])

# Initialize the figure
fig, ax = plt.subplots(figsize=(10,3))

norm = matplotlib.colors.Normalize(vmin=temp_ranges[0][0],
                                   vmax=temp_ranges[-1][1])

ages = get_age_ranges(age_ranges)

####################
# Plot the MCMC fits
####################
for i, fn in enumerate(fits):
    dat = np.load(fn)
    label = fn.split('T')[-1].split('.')[0][:-3]

    t = (temp_ranges[i][0] + temp_ranges[i][1])/2.0

    rgba_color = teff_cbar.reversed()(norm(t))
    hexnum = matplotlib.colors.rgb2hex(rgba_color)

    # Ignores groupings with not enough flares for a good FFD fit
    #if fn == 'slope_fits_T3850-4440_v3.npy':
    #    q = np.array([0,1,3,4,5,6],dtype=int)
    #elif fn == 'slope_fits_T5270-5930_v3.npy':
    #    q = np.arange(1,7,1,dtype=int)
    #else:
    q = np.arange(0,7,1,dtype=int)

    # Plot
    plt.errorbar(ages[:,0][q],
                 dat[0][:,0][q],
                 xerr=[ages[:,1][q], ages[:,2][q]],
                 yerr=[dat[0][:,1][q], dat[0][:,2][q]],
                 marker='o', linestyle='', color=hexnum,
                 label=label+'K', zorder=-i)

#####################################
# Plot data from Ilin et al. (2020) #
#####################################
imarker = 's'
hyades_slopes = -np.array([2.11, 1.89, 1.94, 2.17, 1.99, 1.96])+1
plt.errorbar(695, np.nanmedian(hyades_slopes),
             xerr=75, yerr=np.nanstd(hyades_slopes),
             color='k', marker=imarker)

pleiades_slopes = -np.array([2.13, 2.06, 2.06, 1.99, 2.32, 1.92, 1.91]) + 1
plt.errorbar(127.4,
             np.nanmedian(pleiades_slopes),
             xerr=8, yerr=np.nanstd(pleiades_slopes),
             color='k', marker=imarker)

praesepe_slopes = -np.array([2.09, 2.0, 1.95, 1.89, 2.39, 1.86, 1.91]) + 1
plt.errorbar(617, np.nanmedian(praesepe_slopes), yerr=np.nanstd(praesepe_slopes),
             color='k', marker=imarker, label='Ilin et al. (2020)')

##########################################
# Plot data from Feinstein et al. (2022) #
##########################################
plt.errorbar(np.full(5, 2000),
             -np.array([1.124, 1.408, 1.408, 1.394, 1.319]),
             np.array([0.058, 0.117, 0.121, 0.285, 0.364]),
             ecolor='k', markeredgecolor='k',
             color='w', marker='o', linestyle='', label='Feinstein et al. (2022)')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,
           markerscale=2, fontsize=12)

plt.xscale('log')
plt.ylim(-1.8, 0.0)
plt.xlabel('Age [Myr]', fontsize=20)
plt.ylabel('Flare Frequency\nDistribution Slope, ' + r'$\alpha$', fontsize=20)

ax.set_rasterized(True)

#plt.savefig('/Users/belugawhale/Desktop/mcmc_results.png', dpi=300,
#            bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(figures_dir,'mcmc_results.pdf'),
            dpi=300, bbox_inches='tight')
