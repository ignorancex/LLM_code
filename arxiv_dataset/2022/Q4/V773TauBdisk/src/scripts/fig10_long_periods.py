# Dario Gonzalez Picos
# picos@mail.strw.leidenuniv.nl
# Last update June 2022

'''
--- Script to generate figure 10 of the paper ---
Display the Lomb-Scargle periodogram in the region of long periods (>10 days)
and highlight the peak enhacement at ~ 68 day after removing the high-freq signals
'''
import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from v773tau_functions import * # specific functions

#data_dir = '../data/'
#out_dir = '../plots/'

plt.style.use(paths.data / 'texstyle') # my tex template

# =============================================================================
#                      PREPARE DATA
# =============================================================================
# Load ASAS-SN ground-based photometry from original file
skypatrol_file = paths.data / 'asassn_skypatrol.csv'
df_sp = load_skypatrol(skypatrol_file, show=False) # output labeled in ASAS-SN-g and ASAS-SN-V

# Read the KELT data from the full lightcurve
df_full = pd.read_csv(paths.data / 'v773tau_complete_lightcurve.csv')
df_kelt = df_full[df_full.survey=='KELT']


df = pd.concat([df_sp, df_kelt], join='inner', ignore_index=True)


# =============================================================================
#                    Subtract stellar variability model
#                   and plot periodograms for each survey
# =============================================================================
surveys = ['ASAS-SN-V','ASAS-SN-g','KELT']
n_sines = 6 # number of sine terms to subtract

fig, ax = plt.subplots(len(surveys), figsize=(5,5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.01)
for i,s in enumerate(surveys):
    dfs = df[df.survey==s]
    dfs = dfs.loc[dfs.eclipse==False] # ignore eclipse points
    ax[i] = subtracted_periodogram(dfs, n=n_sines, ax=ax[i]) # wrapper function 

ax[len(ax)-1].set_xlabel('Period [d]')
# plt.show()
fig.savefig(paths.figures / 'fig10_longperiod_periodogram.pdf', dpi=300, bbox_inches='tight')













