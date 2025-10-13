"""
Created on Wed Jul 6 2025

@author: afeinstein20
"""

import matplotlib
import numpy as np
import datetime as dt
from astropy.time import Time
from astropy.table import Table
import matplotlib.pyplot as plt

# Setup the style
params = Table.read('./src/data/rcParams.txt', format='csv')
for i in range(len(params)):
    plt.rcParams[params['name'][i]] = params['value'][i]
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True

# Load the LCO data
LC1 = np.vstack([np.load('./src/data/LCO_1.npy'),
                 np.load('./src/data/LCO_2.npy')])

# Load the TRAPPIST data
trap = Table.read('./src/data/photom-soustraction-3I_TRAPPIST.obs', format='ascii')
LC2 = np.array([trap['epoch'], trap['mag'], trap['err_abs']]).T
print(LC2)

# Load the ATLAS data
LC3 = np.load('./src/data/ATLAS.npy')

# Load the TJO data
LC4 = np.load('./src/data/TJO.npy')

# Errorbar plotting function
def plot(LC, ax, dic):
    time = Time(LC[:,0], format='jd', scale='utc')
    utc = time.to_datetime()
    ax.errorbar(utc, LC[:,1], yerr=LC[:,2],
                markeredgecolor='k', linestyle='', zorder=10,
                markeredgewidth=0.4, linewidth=1, **dic)
    return

# Create the figure
fig, axs = plt.subplots(ncols=4, nrows=1, tight_layout=False, figsize=(18,4.5),
                        gridspec_kw={'wspace':0.1})
#fig.set_facecolor('w')

# Set the start and end dates for each subplot
start_dates = [dt.datetime(2025, 7, 2,6,0,0),dt.datetime(2025, 7, 2,22,0,0),
               dt.datetime(2025, 7, 3,20,0,0),dt.datetime(2025,7,4,20,0,0)]

end_dates   = [dt.datetime(2025, 7, 2,14,0,0),dt.datetime(2025, 7, 3,13,0,0),
               dt.datetime(2025, 7, 4,13,0,0),dt.datetime(2025,7,5,14,0,0)]

#Set date format for x axis
date_format = matplotlib.dates.DateFormatter('%H:%M')

for i in range(len(axs)):

    # plot TRAPPIST light curve
    plot(LC2, axs[i], {'color':'#f6aa1c', 'label':'TRAPPIST\n[$R_C$]',
                       'marker':'o', 'ms':7})

    # plot LCO light curve
    plot(LC1, axs[i], {'color':'#941b0c', 'label':'LCO\n[SDSS r′]',
                      'marker':'^', 'ms':8})

    # plot ATLAS Light curve
    plot(LC3, axs[i], {'color':'#220901', 'label':'ATLAS\n[$o \sim r+i$]',
                       'marker':'s', 'ms':7})

    # plot TJO Light curve
    plot(LC4, axs[i], {'color':'#669bbc', 'label':'TJO\n[SDSS r′]',
                       'marker':'>', 'ms':7})

    axs[i].set_xlim(start_dates[i], end_dates[i])
    axs[i].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=2))
    axs[i].xaxis.set_major_formatter(date_format)
    axs[i].tick_params(axis='x', labelrotation=45)

# Legend
axs[1].legend(bbox_to_anchor=(-0.1, 1.05, 2.3, .102), loc=3,
              ncol=4, mode="expand", borderaxespad=0.,
              markerscale=2, fontsize=16)

# Make the little strikes between subplots
#   and remove some of the subplot axes
d = .025

for j in range(len(axs)):

    kwargs = dict(transform=axs[j].transAxes, color='k', clip_on=False)

    ## Middle subpanels
    if j < len(axs)-1 and j > 0:
        axs[j].spines['right'].set_visible(False)
        axs[j].spines['left'].set_visible(False)


        axs[j].plot((1-d,1+d), (-d,+d), **kwargs)
        axs[j].plot((1-d,1+d),(1-d,1+d), **kwargs)
        axs[j].plot((-d,+d), (1-d,1+d), **kwargs)
        axs[j].plot((-d,+d), (-d,+d), **kwargs)


    ## First subpanel per row
    elif j == 0:
        axs[j].spines['right'].set_visible(False)
        axs[j].plot((1-d,1+d), (-d,+d), **kwargs)
        axs[j].plot((1-d,1+d),(1-d,1+d), **kwargs)

    ## Last subpanel per row
    else:
        axs[j].spines['left'].set_visible(False)
        axs[j].set_yticks([])

        axs[j].plot((-d,+d), (1-d,1+d), **kwargs)
        axs[j].plot((-d,+d), (-d,+d), **kwargs)

# Add the grid to each subplot. Needed to do this to make sure
#   it was consistent
for a in axs:
    a.set_ylim(18.4, 17.4)
    a.set_yticks(np.flip(np.arange(17.4, 18.4, 0.2)))
    for j in np.arange(17.4, 18.4, 0.2):
        a.grid(alpha=0.3, color='gray')
        a.axhline(j, color='gray', alpha=0.3, lw=0.5)

for a in axs[1:]:
    a.set_yticks([])
    a.set_yticklabels([])

# Set the x axis labels
axs[0].set_xlabel('UTC ' + start_dates[0].strftime('%d %b') + ' [h]')
axs[1].set_xlabel('UTC ' + start_dates[1].strftime('%d %b') + ' - ' + end_dates[1].strftime('%d %b') + ' [h]')
axs[2].set_xlabel('UTC ' + start_dates[2].strftime('%d %b') + ' - ' + end_dates[2].strftime('%d %b') + ' [h]')
axs[3].set_xlabel('UTC ' + start_dates[3].strftime('%d %b') + ' - ' + end_dates[3].strftime('%d %b') + ' [h]')

for a in axs:
    a.set_rasterized(True)

axs[0].set_ylabel('Magnitude', fontsize=18)
plt.subplots_adjust(wspace=0.0)

plt.savefig('./src/tex/figures/lightcurve_interstellar.png',
            bbox_inches='tight', dpi=600, transparent=True)
