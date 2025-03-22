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

galex_tab = Table.read(os.path.join(data_dir, 'galex_xmatch.csv'),
                       format='csv')

""" Defining the subtables
Cuts were defined by the HAZMAT team papers.
"""
nuv = galex_tab[(galex_tab['distance_arcsec'] <= 10.0) &
                (galex_tab['nuv_artifact']==0) &
                (galex_tab['nuv_mag'] >= 15.0)]

fuv = galex_tab[(galex_tab['distance_arcsec'] <= 10.0) &
                (galex_tab['fuv_artifact']==0) &
                (galex_tab['fuv_mag'] >= 15.0)]

""" Defining the color maps. """
ccmap = get_galex_colormap(channel='nuv')
pcmap = get_galex_colormap()

""" Making the plot. """
fig, ((ax4, ax3),
      (ax2, ax1),
      (ax6, ax5)) = plt.subplots(ncols=2, nrows=3,
                                 figsize=(12,10),
                                 gridspec_kw={'height_ratios':[0.07,1,1]})
fig.set_facecolor('w')

""""""""""""""""""
""" AX1 -- NUV """
""""""""""""""""""
vmin, vmax = -4, -2
cmap = ccmap
c = np.log10(nuv['nuv_flux_density']/nuv['J_flux_density'])


""" M stars """
q = (nuv['teff'] < 3500) & (nuv['age'] < 50)
ax1.scatter(nuv['R0'][q],
            nuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='o')

q = (nuv['teff'] < 3500) & (nuv['age'] >= 50)
ax5.scatter(nuv['R0'][q],
            nuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='o')

""" K stars """
q = (nuv['teff'] >= 3500) & (nuv['teff'] < 5000) & (nuv['age'] < 50)
ax1.scatter(nuv['R0'][q],
            nuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='^')

q = (nuv['teff'] >= 3500) & (nuv['teff'] < 5000) & (nuv['age'] >= 50)
ax5.scatter(nuv['R0'][q],
            nuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='^')

""" G stars """

q = (nuv['teff'] > 5000) & (nuv['age'] < 50)
im = ax1.scatter(nuv['R0'][q],
                 nuv['flare_rate'][q],
                 c=c[q],
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 marker='s')

q = (nuv['teff'] > 5000) & (nuv['age'] >= 50)
im = ax5.scatter(nuv['R0'][q],
                 nuv['flare_rate'][q],
                 c=c[q],
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 marker='s')

cbar = plt.colorbar(im, cax=ax3, orientation='horizontal')
cbar.set_ticks(np.arange(-4, -1, 1))
ax3.set_title(r'$log_{10}(f_{NUV}/ f_J$)', fontsize=16)
ax3.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

""""""""""""""""""""""""
""" AX 2 -- FUV FLUX """
""""""""""""""""""""""""
cmap=pcmap
vmin, vmax = -5, -2
c = np.log10(fuv['fuv_flux_density']/fuv['J_flux_density'])

""" < stars """

q = (fuv['teff'] < 3500) & (fuv['age'] < 50)
ax2.scatter(fuv['R0'][q],
            fuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='o')

q = (fuv['teff'] < 3500) & (fuv['age'] >= 50)
ax6.scatter(fuv['R0'][q],
            fuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='o')

""" K stars """

q = (fuv['teff'] >= 3500) & (fuv['teff'] < 5000) & (fuv['age'] < 50)
ax2.scatter(fuv['R0'][q],
            fuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='^')

q = (fuv['teff'] >= 3500) & (fuv['teff'] < 5000) & (fuv['age'] >= 50)
ax6.scatter(fuv['R0'][q],
            fuv['flare_rate'][q],
            c=c[q],
            vmin=vmin, vmax=vmax, cmap=cmap,
            marker='^')

""" G stars """

q = (fuv['teff'] > 5000) & (fuv['age'] < 50)
im = ax2.scatter(fuv['R0'][q],
                 fuv['flare_rate'][q],
                 c=c[q],
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 marker='s')

q = (fuv['teff'] > 5000) & (fuv['age'] >= 50)
im = ax6.scatter(fuv['R0'][q],
                 fuv['flare_rate'][q],
                 c=c[q],
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 marker='s')

cbar = plt.colorbar(im, cax=ax4, orientation='horizontal')
ax4.set_title(r'$log_{10}(f_{FUV} / f_J)$', fontsize=16)
ax4.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

ax6.plot(100,100,marker='o',label='M stars', lw=0, color='k')
ax6.plot(100,100,marker='^',label='K stars', lw=0, color='k')
ax6.plot(100,100,marker='s',label='G stars', lw=0, color='k')
ax6.legend(bbox_to_anchor=(0.3, -0.4, 1.4, .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,
           markerscale=2)

for ax in [ax1, ax2, ax5, ax6]:
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(10**-3, 1.5)
    ax.set_xlim(0.001, 2.0)
    ax.axvline(0.15, color='k', zorder=0)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax1.set_yticklabels([])
ax5.set_yticklabels([])


ax6.set_ylabel('Flare Rate [day$^{-1}$]', y=1.1)
ax6.set_xlabel('Rossby Number', x=1.0)

ax7 = ax1.twinx()
ax7.set_ylabel('4.5 - 50 Myr')
ax7.set_yticks([])

ax8 = ax5.twinx()
ax8.set_ylabel('50 - 250 Myr')
ax8.set_yticks([])

plt.subplots_adjust(wspace=0.1, hspace=0.1)



loc, fs = 1.3e-3, 16
ax1.text(s='N$_{stars}$ = '+str(int(len(nuv[(nuv['age']<50) & (nuv['nuv_artifact']==0) &
                                            (nuv['flare_rate']>0)]))),
         x=loc, y=loc, fontsize=fs)
ax5.text(s='N$_{stars}$ = '+str(int(len(nuv[(nuv['age']>=50) & (nuv['nuv_artifact']==0) &
                                            (nuv['flare_rate']>0)]))),
         x=loc, y=loc, fontsize=fs)

ax2.text(s='N$_{stars}$ = '+str(int(len(fuv[(fuv['age']<50) & (fuv['fuv_artifact']==0) &
                                            (fuv['flare_rate']>0)]))),
         x=loc, y=loc, fontsize=fs)
ax6.text(s='N$_{stars}$ = '+str(int(len(fuv[(fuv['age']>=50) & (fuv['fuv_artifact']==0) &
                                            (fuv['flare_rate']>0)]))),
         x=loc, y=loc, fontsize=fs)

plt.savefig(os.path.join(figures_dir,'galex.pdf'),
            dpi=300, bbox_inches='tight')
