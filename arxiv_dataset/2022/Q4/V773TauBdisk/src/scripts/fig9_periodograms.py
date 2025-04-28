
# Dario Gonzalez Picos
# picos@mail.strw.leidenuniv.nl
# Last update June 2022

'''
--- Script to generate figure 9 of the paper ---
Show the Lomb-Scargle periodogram with the main periods for the 
merged ground-based photometry and TESS alone
'''
import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
from v773tau_functions import * # specific functions

def myperiodogram(ls, min_period, max_period, ax=None, n=100, **kwargs):
    # min_period, max_period, peak_points = 1., 5000,  10
    frequencies, power = ls.autopower(minimum_frequency=(1./max_period), 
                                      maximum_frequency=(1/min_period),
                                      samples_per_peak=40)
    # convert to periods
    periods = 1/frequencies
    # extract best peaks
    df_best_peaks = extract_peaks(power, periods, 0.00)
    ax.plot(periods, power,'-', **kwargs)
    for period, height in zip(df_best_peaks[:n].periods, df_best_peaks[:n].height):
        ax.plot(period, height, 'd', ms=4, label='{:.2f} d'.format(period))
    ax.legend(fontsize=10, prop={'size': 9})
    ax.set_xlim(left=1.)
    ax.set_ylim(bottom=0.)
    return df_best_peaks

#data_dir = '../data/'
#out_dir = '../plots/'

plt.style.use(paths.data / 'texstyle') # my tex template

# Load TESS
df_tess = pd.read_csv(paths.data / 'v773tau_eleanor.csv')
df_tess['time'] = df_tess.time + 7000

# Load merged and calibrated ground-based photometry
df = pd.read_csv(paths.data / 'v773tau_complete_lightcurve.csv')

## Generate figure ##
ecl = df.eclipse == True
fig, ax = plt.subplots(2, figsize=(5,4), sharex=True)
plt.subplots_adjust(hspace=0.0)
min_period, max_period = 1.05, 6.
ls = LombScargle(df[~ecl].time, df[~ecl].flux, df[~ecl].flux_err)
plt_dict = {'c':'black', 'alpha':0.85}
df_best_peaks = myperiodogram(ls, min_period, max_period, ax=ax[0], n=6, **plt_dict)    
ax[0].set_ylabel('Power[-]')
ax[0].text(s='Merged photometry',x=0.43,y=0.85, transform=ax[0].transAxes, fontsize=11, color='black')

ls = LombScargle(df_tess.time, df_tess.flux, df_tess.flux_err)
plt_dict = {'c':'brown', 'alpha':0.85}
df_best_peaks = myperiodogram(ls, min_period, max_period, ax=ax[1], n=4, **plt_dict)
ax[1].set(xlabel='Period [d]', ylabel='Power [-]')
ax[1].set_xticks(np.arange(1,6.1,0.5))
ax[1].set_yticks(np.arange(0.00, 0.31, 0.10))
ax[1].set_yticklabels(['0.00', '0.10','0.20','0.30'])
ax[1].text(s='TESS',x=0.54,y=0.85, transform=ax[1].transAxes, fontsize=11, color='brown')

# plt.show()
fig.savefig(paths.figures / 'fig9_periodograms.pdf', dpi=300, bbox_inches='tight')
