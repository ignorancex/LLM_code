import os
import sys
import matplotlib
import numpy as np
from astropy import units
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from plotting_utils import *
from data_utils import *

from paths import (
    figures as figures_dir,
    data as data_dir,
)

set_rcparams()

tab = Table.read(os.path.join(data_dir, 'combined_prot_flarerate_v6.csv'),
                 format='csv')
tab = tab[tab['Good'] == 1]

prot = tab['Prot']
teff = tab['Teff']
age  = tab['age']
fr   = tab['flare_rate']

## Calculate Rossby Number ##
log_tau = 1.16 - 1.49*np.log10(tab['mass'])-0.54*np.log10(tab['mass'])**2
tau = 10**log_tau
R0_1 = prot/tau

## Setup Figure ##
fig, axes = plt.subplots(figsize=(20,8), nrows=2, ncols=4, sharex=True,
                         sharey=True, gridspec_kw={'width_ratios':[1,1,1.2,1.2]})
axes = axes.reshape(-1)
fig.set_facecolor('w')
nbins = 30

## Define binning ##
R0_bins = np.logspace(-2, 0, nbins)
fr_bins = np.logspace(-2.5,0.1,nbins)

## Colormaps ##
c_old = ['#03045eff', '#023e8aff', '#0077b6ff', '#0096c7ff', '#00b4d8ff',
         '#48cae4ff', '#90e0efff', '#ade8f4ff', '#caf0f8ff','#ffffff']
cmap_old = LinearSegmentedColormap.from_list('old', c_old).reversed()

c_young = ['#2f184bff', '#532b88ff', '#9b72cfff', '#c8b1e4ff', '#f4effaff',
           '#ffffff']
cmap_young = LinearSegmentedColormap.from_list('young', c_young).reversed()

c = ['#391c06ff', '#552a09ff', '#71380cff',
     '#8e450eff', '#aa5311ff', '#c66114ff', '#e36f17ff', '#ff7d1aff',
     '#ffba86', '#ffcca4', '#fff3f3', '#ffffff']
cmap_c = LinearSegmentedColormap.from_list('k', c).reversed()


#######################
### M DWARF SUBPLOT ###
#######################
axes[0].set_title('M stars')
q2 = ((np.isfinite(np.log10(fr))==True) & (teff <= 3500) &
       (age <= 50))
im = axes[0].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_young, vmax=5)


q2 = ((np.isfinite(np.log10(fr))==True) & (teff <= 3500) &
      (age > 50) )
im = axes[4].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_old, vmin=0, vmax=8)



#######################
### K DWARF SUBPLOT ###
#######################
axes[1].set_title('K stars')
q2 = ((np.isfinite(np.log10(fr))==True) & (teff > 3500) & (teff < 5000) &
       (age <= 50))
im = axes[1].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_young, vmax=5)


q2 = ((np.isfinite(np.log10(fr))==True) & (teff > 3500) & (teff < 5000) &
      (age > 50))
im = axes[5].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_old, vmin=0, vmax=8)


#######################
### G DWARF SUBPLOT ###
#######################
axes[2].set_title('G stars')
q2 = ((np.isfinite(np.log10(fr))==True) & (teff >= 5000) &
       (age <= 50))
im = axes[2].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_young, vmax=5)
plt.colorbar(im[-1], ax=axes[2], aspect=12)


q2 = ((np.isfinite(np.log10(fr))==True) & (teff >= 5000) &
      (age > 50))
im = axes[6].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_old, vmin=0, vmax=8)
cbar = plt.colorbar(im[-1], ax=axes[6], aspect=12)
cbar.set_ticks(np.arange(0,9,2))

########################
### COMBINED SUBPLOT ###
########################
q2 = ((np.isfinite(np.log10(fr))==True) & (teff >= 2300) & (teff<6000) &
       (age <= 50))
im = axes[3].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_c, vmax=10)
plt.colorbar(im[-1], ax=axes[3], aspect=12)


q2 = ((np.isfinite(np.log10(fr))==True) & (teff >= 2300) & (teff<6000) &
      (age > 50))
im = axes[-1].hist2d(R0_1[q2], fr[q2],
                    bins=[R0_bins, fr_bins],
                    cmap=cmap_c, vmin=0, vmax=10)
cbar = plt.colorbar(im[-1], ax=axes[-1], aspect=12)
cbar.set_ticks(np.arange(0,12,2))


### Setting plot labels and text
cbar.set_label('Number of Stars per bin', labelpad=10, y=1.1)
axes[5].set_xlabel(r'Rossby Number [P$_{rot} / \tau$]', x=1.1)
axes[4].set_ylabel('Flare Rate [day$^{-1}$]', y=1.1)

x=10**-2.4
y=10**-2.4
for i in range(4):
    axes[i].text(s='4.5 - 50 Myr', x=x, y=y, fontsize=16)
for i in range(4,8):
    axes[i].text(s='50 - 250 Myr', x=x, y=y, fontsize=16)

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_ylim(10**-2.5, 1)
axes[0].set_xlim(10**-2.5, 1)

axes[0].set_title('M stars')
axes[1].set_title('K stars')
axes[2].set_title('G stars')
axes[3].set_title('GKM stars combined')

plt.subplots_adjust(wspace=0.2, hspace=0.1)

#plt.savefig('/Users/belugawhale/Desktop/prot_histograms.png', dpi=300,
#            bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(figures_dir, 'prot_histograms.pdf'), dpi=300,
            bbox_inches='tight')
