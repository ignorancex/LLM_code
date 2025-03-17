import os
import sys
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paths import (
    figures as figures_dir,
    data as data_dir,
    scripts as script_dir
)

from plotting_utils import *

set_rcparams()

sample = Table.read(os.path.join(data_dir,'moca_sample.csv'), format='csv')
_, uni = np.unique(sample['ID'], return_index=True)
sample = sample[uni]

RA, Dec = sample['ra'], sample['dec']
org = 0
projection = 'mollweide'

parula = age_colorbar()

x = np.remainder(RA+360-org,360) # shift RA values
ind = x>180
x[ind] -=360    # scale conversion to [-180, 180]
x=-x    # reverse the scale: East to the left
tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
tick_labels = np.remainder(tick_labels+360+org,360)

fig, axes = plt.subplots(figsize=(8, 9), nrows=2,
                      gridspec_kw={'height_ratios':[2,1]})
axes[0].set_visible(False)
axes[1].set_visible(False)

ax = fig.add_subplot(211, projection=projection)
img = ax.scatter(np.radians(x),np.radians(Dec), c=np.log10(sample['age']),
                 vmin=np.log10(5), vmax=np.log10(250), s=15, cmap=parula)
ax.set_xticklabels(tick_labels, fontsize=18)     # we add the scale on the x axis

ax.set_xlabel("RA", fontsize=20)
ax.set_ylabel("Dec", fontsize=20)

ax.grid(True)
cbar = plt.colorbar(img, location='bottom', shrink=0.65)
cbar.ax.set_xlabel('Age [Myr]', fontsize=20)

cbar.set_ticks(np.linspace(np.log10(5), np.log10(250), 7))
cbar.ax.set_xticklabels(np.linspace(5,250,7,dtype=int), fontsize=14)
ax.set_rasterized(True)

## histogram of Teff ##
ax2 = fig.add_subplot(212)
ax2.hist(sample['Teff'], bins=np.linspace(2500,6000,100), color='k',
         alpha=0.4)
ax2.hist(sample['Teff'], bins=np.linspace(2500,6000,100), color='k',
         histtype='step', lw=2)
ax2.set_xlim(2500,6000)
ax2.set_rasterized(True)
ax2.set_xlabel('$T_{eff}$ [K]')
ax2.set_ylabel('$N_{stars}$')


plt.savefig(os.path.join(figures_dir,'sample.pdf'), dpi=300,
            bbox_inches='tight')
