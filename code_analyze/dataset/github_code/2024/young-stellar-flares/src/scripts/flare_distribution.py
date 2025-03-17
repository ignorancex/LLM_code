import os
import sys
import numpy as np
from astropy import units
from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting_utils import *

from paths import (
    figures as figures_dir,
    data as data_dir,
)

set_rcparams()

tab = Table.read(os.path.join(data_dir,
                              'final_flare_catalog_v6.csv'),
                 format='csv')
tab = tab[(tab['ed'] > 0) & (tab['age']>0)]

sample = Table.read(os.path.join(data_dir,'moca_sample.csv'), format='csv')

#fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6,18))
fig, (ax0, ax2) = plt.subplots(nrows=2, figsize=(6,12))
fig.set_facecolor('w')

ed = (tab['ed']*units.min).to(units.s)
erg = tab['flare_energy']

q = ((np.isnan(tab['teff']) == False) & (tab['teff'] > 0) &
     (tab['teff'] <= 6000) & (erg.value > 0) & (ed.value > 0))

#================== AX0 ========================#
age = age_colorbar()

uni, args = np.unique(sample['moca_aid'], return_index=True)
argsort = np.argsort(sample['age'][args])

counts = np.zeros((len(uni),3))

for i, grp in enumerate(uni[argsort]):

    count = len(sample[(sample['moca_aid']==grp)])

    cflares = len(tab[(tab['moca_aid']==grp) & (tab['teff']<=6000) &
                      (np.isnan(tab['flare_energy']) == False)])

    counts[i][0] = count
    counts[i][1] = cflares
    counts[i][2] = sample['age'][args][argsort][i]

img = ax0.scatter(counts[:,0], counts[:,1], c=np.log10(counts[:,2]),
                  vmin=np.log10(4), vmax=np.log10(250), s=120,
                  cmap=age, edgecolor='k', lw=0.75)
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.08)

cbar = plt.colorbar(img, label='Age [Myr]', cax=cax)
ticks = np.round(10**np.linspace(np.log10(4), np.log10(250), 6))
cbar.set_ticks(np.log10(ticks))
cbar.set_ticklabels(np.array(ticks,dtype=int))

ax0.set_ylabel('$N_{flares}$', fontsize=24)
ax0.set_xlabel('$N_{stars}$', fontsize=24)
ax0.set_yscale('log')
ax0.set_xscale('log')

#================== AX1 ========================#
"""
prob = prob_colorbar()

img = ax1.scatter(np.log10(erg.value[q]), np.log10(ed.value[q]),
                  c=tab['prob'][q]*100, cmap=prob,
                  vmin=75, vmax=100, s=5)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.08)

plt.colorbar(img, label='Probability [%]', cax=cax)

ax1.set_xlabel(r'log$_{10}$(Flare Energy [erg])')
ax1.set_ylabel(r'log$_{10}$(ED [sec])')
"""
#================== AX2 ========================#
teff = teff_colorbar()

img = ax2.scatter(np.log10(erg[q].value), np.log10(ed[q].value),
                  c=np.log10(tab['teff'][q]),
                  vmin=3.4, vmax=3.9, s=5, cmap=teff)

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.08)
plt.colorbar(img, label='log$_{10}$(T$_{eff}$ [K])', cax=cax)

ax2.set_xlabel(r'log$_{10}$(Flare Energy [erg])')
ax2.set_ylabel(r'log$_{10}$(ED [sec])')

for ax in [ax2]:#, ax1]:
    ax.set_xlim(27 ,35)
    ax.set_ylim(-1.65,4.2)

ax0.set_rasterized(True)
#ax1.set_rasterized(True)
ax2.set_rasterized(True)

plt.subplots_adjust(hspace=0.3)

plt.savefig(os.path.join(figures_dir,'flare_distribution.pdf'),
            bbox_inches='tight',
            dpi=300)
