import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from astropy.table import Table, Column, vstack
from astropy.time import Time
from altaipony.ffd import FFD
from scipy.optimize import curve_fit

from plotting_utils import *

from paths import (
    figures as figures_dir,
    data as data_dir,
)

from cycle_utils import *

set_rcparams()

# Targets with evidence of stellar cycles
evidence = np.array([142015852.0, 270676943.0, 272349442.0,
                     308186412.0, 391745863.0, 393490554.0,
                     452357628.0, 235056185, 260351540, 339668420, 350559457])

# Load in TESS data
tab = Table.read(os.path.join(data_dir,
                              'final_flare_catalog_v6.csv'),
                 format='csv')
tab = tab[(tab['ed'] > 0) & (tab['age']>0)]

# Load in solar data
rhessi = Table.read(os.path.join(data_dir, 'rhessi_reformatted.csv'), format='csv')
rhessi = rhessi[(rhessi['X_pos']!=0) & (rhessi['Y_pos']!=0) &
          (rhessi['energy_keV_low']>3) & (rhessi['energy_keV_upp']>6)]

# Getting colors
url = '001219-003946-005f73-0a9396-4fb3aa-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226-5a0002'
cmap = ['#'+i for i in url.split('-')]

# Sets up the figure and the correct number of subplots
fig = plt.figure(figsize=(50, 14))
fig.set_facecolor('w')

gs0 = GridSpec(2, 1, figure=fig, height_ratios=[2.1,1], hspace=0.3)

gs00 = GridSpecFromSubplotSpec(2, len(evidence)+1, subplot_spec=gs0[0], hspace=0.1,
                               wspace=0.4)
gs01 = GridSpecFromSubplotSpec(1, len(evidence)+1, subplot_spec=gs0[1], wspace=0.4)

axes = []
for i in range((len(evidence)+1)*2):
    axes.append(fig.add_subplot(gs00[i]))
for i in range(len(evidence)+1):
    axes.append(fig.add_subplot(gs01[i]))

#########################
### PLOTS FOR THE SUN ###
#########################

total_peryear = np.zeros((16,2))
per_year = np.arange(rhessi['peak_mjd'][0],rhessi['peak_mjd'][-1], 365)
solar_xi, solar_peaks, tmid = [], [], []

for i in range(len(per_year)-4):

    temp = rhessi[(rhessi['peak_mjd']>= per_year[i]) &
               (rhessi['peak_mjd'] < per_year[i+1]) &
               (rhessi['energy_keV_low'] >= 25.0) &
               (np.isnan(rhessi['flare_e_approx'])==False) &
               (np.isfinite(rhessi['flare_e_approx'])==True)]

    arg = np.argmax(temp['peak_counts'])
    avg = (per_year[i]+per_year[i+1])/2.0
    avg = (avg+2400000)-2457000

    axes[0].plot(avg, np.log10(np.nansum(temp['dur'])/1e-3),
                 'o', markeredgecolor='k',
                 zorder=30, ms=10, color=cmap[i])
    solar_xi.append(np.log10(np.nansum(temp['dur'])/1e-3))

    axes[len(evidence)+1].plot(avg, len(temp['peak_counts'])/365.0,
                               'o', markeredgecolor='k',
                               zorder=30, ms=10, color=cmap[i])
    solar_peaks.append(len(temp['peak_counts'])/365.0)
    tmid.append(avg)

    ed, freq = cumulative_ffd(temp, ed_key='dur', eng_key='peak_counts')
    axes[(len(evidence)+1)*2].plot(ed, freq, color=cmap[i], lw=3)

solar_xi = np.array(solar_xi)
solar_peaks = np.array(solar_peaks)

fit_xi = curve_fit(sine_wave, tmid, solar_xi,
                   bounds=((-100,365*5., -100, -100),
                           (100,365*30.,100,100)))
axes[0].plot(np.linspace(tmid[0],tmid[-1], 100),
             sine_wave(np.linspace(tmid[0],tmid[-1], 100), fit_xi[0][0],
                       fit_xi[0][1], fit_xi[0][2], fit_xi[0][3]),
             color='k', zorder=0)

fit_peak = curve_fit(sine_wave, tmid, solar_peaks,
                     p0=fit_xi[0],
                     bounds=((-100,365*5., -100, -100),
                             (100,365*30,100,100)))
axes[len(evidence)+1].plot(np.linspace(tmid[0],tmid[-1], 100),
             sine_wave(np.linspace(tmid[0],tmid[-1], 100), fit_peak[0][0],
                       fit_peak[0][1], fit_peak[0][2], fit_peak[0][3]),
             color='k', zorder=0)

axes[(len(evidence)+1)*2].set_yscale('log')
axes[(len(evidence)+1)*2].set_xscale('log')
axes[(len(evidence)+1)*2].set_xticks(np.logspace(1,7,4))
axes[0].set_title('The Sun', fontweight='bold', fontsize=18)
axes[(len(evidence)+1)*2].set_xlabel('Peak Counts')
axes[(len(evidence)+1)*2].set_xlabel('Peak Counts')
axes[0].set_xticklabels([])
axes[0].set_yticks(np.round(np.linspace(solar_xi.min(), solar_xi.max(), 3), 1))
axes[len(evidence)+1].set_yticks(np.round(np.linspace(solar_peaks.min(),
                                                      solar_peaks.max(), 3), 1))
for ax in [axes[0], axes[len(evidence)+1]]:
    ax.set_xticks([-4000, -2000, 0])
axes[len(evidence)+1].set_xticklabels([-4000, -2000, 0], fontsize=18)


x = 1
offset = len(evidence)+1

#############################
### PLOTS FOR OTHER STARS ###
#############################
colors = [2, 6, 10]

for i, star in enumerate(evidence):
    files = np.sort([i for i in os.listdir(os.path.join(data_dir, 'stellar_cycle_lcs')) if str(int(star)) in i])
    time = np.array([])

    for f, fn in enumerate(files):
        dat = np.load(os.path.join(os.path.join(data_dir, 'stellar_cycle_lcs'), fn))
        time = np.append(time, dat[0])
        del dat

    q = np.isnan(time) == False
    time = time[q]

    midtime, xi, peaks = np.array([]), np.array([]), np.array([])
    print(star)
    for j in range(3):
        # Set the time limit in BJD - 2457000
        if j == 0: lim=[0,1800]
        elif j == 1: lim=[1800,2600]
        else: lim=[2600,4000]

        tot = (len(time[(time < lim[1]) & (time > lim[0])] * 2)*units.min).to(units.day)

        year = tab[(tab['Target_ID']==star) & (tab['tpeak'] > lim[0]) &
                    (tab['tpeak'] < lim[1]) & (tab['prob'] >= 0.99)]

        # Calculate annual fractional luminosity
        ind_xi, t_xi = calculate_xi(year, time, lim, cmap, axes[x], colors[j])

        # Calculate annual flare rate
        ind_fr, _ = calculate_fr(year, time, lim, cmap, axes[x+offset], colors[j])

        year_max = tab[(tab['Target_ID']==star) & (tab['tpeak'] > lim[0]) &
                    (tab['tpeak'] < lim[1])]
        year_min = tab[(tab['Target_ID']==star) & (tab['tpeak'] > lim[0]) &
                    (tab['tpeak'] < lim[1]) & (tab['prob']>=0.995)]

        xi = np.append(xi, ind_xi)
        peaks = np.append(peaks, ind_fr)
        midtime = np.append(midtime, t_xi)

        # plots the cumulative flare frequency distribution
        ed, freq = cumulative_ffd(year)
        axes[x+offset*2].plot(ed, freq, color=cmap[colors[j]], lw=3)

        ed_max, freq_max = cumulative_ffd(year_max)
        ed_min, freq_min = cumulative_ffd(year_min)

        rebin_min, rebin_max = np.zeros(len(ed)), np.zeros(len(ed))
        for e in range(len(ed)):
            argmin = np.argmin(np.abs(ed[e]-ed_max))
            rebin_max[e] = freq_max[argmin]

            argmin =np.argmin(np.abs(ed[e]-ed_min))
            rebin_min[e] = freq_min[argmin]

        axes[x+offset*2].fill_between(ed, rebin_min, rebin_max, zorder=0,
                                      alpha=0.4, color=cmap[colors[j]],
                                      lw=0)

    newx = np.linspace(midtime.min(), midtime.max(), 100)
    q1 = np.isfinite(xi) == True

    if len(xi[q1]) == 3:
        deg = 2
    else:
        deg = 1

    fit_xi = np.polyfit(midtime[q1], xi[q1], deg=deg)
    func1 = np.poly1d(fit_xi)
    axes[x].plot(newx, func1(newx), color='k', zorder=0)

    q = np.isfinite(peaks) == True
    fit_pk = np.polyfit(midtime[q], peaks[q], deg=deg)
    func2 = np.poly1d(fit_pk)
    axes[x+offset].plot(newx, func2(newx), color='k', zorder=0)


    # This is all subplot aesthetic stuff
    for ax in [axes[x], axes[x+offset]]:
        ax.set_xticks([1600, 2300, 3000])
        ax.set_xticklabels([1600, 2300, 3000], fontsize=18)

    if x < (len(evidence)+1):
        axes[x].set_xticklabels([])

    yticks = np.round(np.linspace(xi[q1].min(), xi[q1].max(), 3),2)
    axes[x].set_yticks(yticks)
    axes[x].set_yticklabels(yticks, fontsize=18)

    yticks = np.round(np.linspace(peaks[q].min(), peaks[q].max(), 3),2)
    axes[x+offset].set_yticks(yticks)
    axes[x+offset].set_yticklabels(yticks, fontsize=18)

    axes[x].set_title('TIC  {}'.format(int(star)), fontsize=18, fontweight='bold')

    axes[x].set_rasterized(True)
    axes[x+offset].set_rasterized(True)
    axes[x+offset*2].set_rasterized(True)
    x += 1
    # This ends the subplot aesthetic stuff

axes[(len(evidence)+1)+6].set_xlabel('Time [BJD - 2457000]', fontsize=24)

for i in range((len(evidence)+1)*2+1, (len(evidence)+1)*3):

    axes[i].set_yscale('log')
    axes[i].set_xscale('log')
    axes[i].set_xticks(np.logspace(29,33,3))
    axes[i].set_xticklabels(['$10^{29}$', '$10^{31}$', '$10^{33}$'])
    axes[i].set_ylim(0.002, 0.4)

    if i > (len(evidence)+1)*2+1:
        axes[i].set_yticklabels([])
    if i == (len(evidence)+1)*2+6:
        axes[i].set_xlabel('Flare Energy [erg]', fontsize=24)

axes[0].set_ylabel(r'log$_{10}(\xi_{flare}/t_{exp})$')
axes[len(evidence)+1].set_ylabel('log$_{10}$(Flare\nRate [day$^{-1}$])')
axes[(len(evidence)+1)*2].set_ylabel('FFD [day$^{-1}$]')

plt.subplots_adjust(wspace=0.3, hspace=0.4)

plt.savefig(os.path.join(figures_dir, 'stellar_cycles.pdf'),
            dpi=300, bbox_inches='tight')
