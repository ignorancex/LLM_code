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

cmap = teff_colorbar().reversed()

##################################################
### Load tables and make appropriate data cuts ###
##################################################
flares = Table.read(os.path.join(data_dir,'planet_host_rates.csv'), format='csv')
tab = Table.read(os.path.join(data_dir,'final_flare_catalog_v6.csv'),
                 format='csv')
tab = tab[(tab['ed'] > 0) & (tab['age']>0)]

#############################
### Calculate flare rates ###
#############################
yp = np.zeros(len(np.unique(flares['Target_ID'])))
names =  np.zeros(len(np.unique(flares['Target_ID'])), dtype='U30')

for i, unique in enumerate(np.unique(flares['Target_ID'])):
    tt = flares[flares['Target_ID']==unique]['tot_time_obs'][0]
    name = flares[flares['Target_ID']==unique]['name'][0]
    nflare = np.nansum(flares[flares['Target_ID']==unique]['prob'])
    yp[i] = np.round(nflare/tt,3)
    names[i] = name

##########################################
### Append targets which have 0 flares ###
##########################################
yp = np.insert(yp, 4, 0)
yp = np.append(yp, np.zeros(3))

names = np.insert(names, 4, 'HD 109833')
names = np.append(names, ['HD 18599', 'TOI 2076', 'HD 110082'])

##########################################
### Ages and Teff of planet host stars ###
##########################################
ages = [11, 17, 22, 23, 27, 36, 36,
        45, 50, 120, 133, 133, 180,
        200, 204, 250]
teff = [3055, 5767, 3588, 5159.64,
        5881, 4324, 5241, 5414, 4945,
        5481, 5991, 5668.04, 5163, 6413,
        4730, 5222]

###################################################
### Calculate flare rates for equivalent sample ###
###################################################
avg_rate = np.zeros(len(ages))
std_rate = np.zeros((len(ages),2))
for i in range(len(ages)):
    temp = tab[(tab['age'] >= ages[i] - 30) & (tab['age'] <= ages[i] + 30) &
               (tab['teff'] >= teff[i] - 1000) & (tab['teff'] <= teff[i] + 1000)]
    unique = np.unique(temp['Target_ID'])
    rates = np.zeros(len(unique))

    for j, target in enumerate(unique):
        tt = temp[temp['Target_ID']==target]['total_obs_time'][0]
        n = np.nansum(temp[temp['Target_ID']==target]['prob'])
        rates[j] = n/tt

    avg_rate[i] = np.nanmedian(rates)
    std_rate[i][0] = np.percentile(rates, 16)
    std_rate[i][1] = np.percentile(rates, 84) - np.nanmedian(rates)

#######################
### Create the plot ###
#######################
norm = matplotlib.colors.Normalize(vmin=2300,vmax=5930)

fig, axes = plt.subplots(nrows=2, figsize=(10,6),
                               gridspec_kw={'height_ratios':[1,2]})
ax1, ax2 = axes.reshape(-1)

# For the color bar only
sc = ax1.scatter(np.arange(len(yp)), yp, c=teff, cmap=cmap, vmin=2300, vmax=5930, s=0)

for i in range(len(teff)):
    rgba_color = cmap.reversed()(norm(teff[i]))
    hexnum = matplotlib.colors.rgb2hex(rgba_color)

    if i == 2:
        ax = ax1
    else:
        ax = ax2

    ax2.vlines(i, std_rate[:,0][i], std_rate[:,1][i], lw=10, alpha=0.5,
               color=hexnum)
    ax.plot(i, yp[i], 'o', color=hexnum, markeredgecolor='k', lw=2,
             ms=12)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.set_xticks([])

d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=2.5)
ax1.plot((-d, +d), (-d-0.03, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d-0.03, +d), **kwargs)  # top-right diagonal

d=0.015
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax2.set_ylim(-0.05,0.4)
plt.ylabel('Flare Rate [day$^{-1}$]', y=0.8)
plt.xlabel('Host Star Name')
plt.xticks(np.arange(len(yp)),
           labels=names, rotation='vertical', fontsize=14)

ax1.set_xticks([])
ax1.set_xlim(-0.8,len(yp))

ax2.set_yticks(np.arange(0,0.5,0.1))
ax1.set_yticks(np.arange(2.2,2.3,0.05))
ax1.set_ylim(2.15,2.25)

plt.subplots_adjust(hspace=0.1)

cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
cbar.set_label('T$_{eff}$ [K]')

plt.savefig(os.path.join(figures_dir,'yp_rates.pdf'),
            dpi=300, bbox_inches='tight')
