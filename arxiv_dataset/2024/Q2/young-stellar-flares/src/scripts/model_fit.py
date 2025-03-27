import os
import sys
import numpy as np
from astropy import units
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import flare_models as fm
from plotting_utils import *

from paths import (
    figures as figures_dir,
    data as data_dir,
)

set_rcparams()

# load the light curve data to plot
def load_lightcurve(tic):
    # load in the file names
    path = os.path.join(data_dir,'TESS')

    directory = os.listdir(path)
    tempfiles = np.sort([i for i in directory if str(int(tic)) in i and
                         i.endswith('stella.npy')])

    x, y, e, p = np.array([]), np.array([]), np.array([]), np.array([])

    for j in range(len(tempfiles)):
        dat = np.load(os.path.join(path,tempfiles[j]),
                      allow_pickle=True)
        q = ((np.isfinite(dat[0])==True) & (np.isfinite(dat[1])==True) &
             (np.isnan(dat[0])==False) & (np.isnan(dat[1])==False))

        x = np.append(x, dat[0][q])
        y = np.append(y, dat[1][q])
        e = np.append(e, dat[2][q]/np.nanmedian(dat[1][q]))
        p = np.append(p, dat[3][q])

    return x, y, e, p

# read in the table with the best-fit flare parameters
tab = Table.read(os.path.join(data_dir,'chi2_test.csv'), format='csv')

# initialize the figure
fig = plt.figure(figsize=(8,14))

gs0 = gridspec.GridSpec(2,1, figure=fig, height_ratios=[0.9,1], hspace=0.35)

ax1 = fig.add_subplot(gs0[0])

# add the flare example subplots
gs01 = gs0[1].subgridspec(3,2, wspace=0.3)

flare_axes = [fig.add_subplot(gs01[0,0]), fig.add_subplot(gs01[0,1]),
              fig.add_subplot(gs01[1,0]), fig.add_subplot(gs01[1,1]),
              fig.add_subplot(gs01[2,0]), fig.add_subplot(gs01[2,1])]


# plot the chi^2 values
ax1.plot(tab['TM22'], tab['P22'], 'o', color='#323031')
line = np.linspace(1e-4, 10, 100)
ax1.plot(line, line, '#ecc8af')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$\chi^2$ (Llamaradas Estelares)')
ax1.set_ylabel('$\chi^2$ (Double-peak)')
ax1.set_xlim(1e-4,10)
ax1.set_ylim(1e-4,10)
ax1.text(1.3*10**-4, 4, '(A)', fontweight='bold')
ax1.set_rasterized(True)

# plot the flare examples
tm_parameters = ['tm_tpeak', 'tm_fwhm', 'tm_ampl']
p_parameters = ['A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']
idx = [780,442, 66, 346, 33, 513]
letter = ['B', 'C', 'D', 'E', 'F', 'G']
offset = [2.1, 0.65, 0.55, 1.05, 1.1, 0.6]

for i in range(len(idx)):
    j = idx[i]

    # load the light curve for the target
    vals = load_lightcurve(int(tab['Target_ID'][j]))

    # limit the x-range around the flare
    q = (vals[0]>tab['tpeak'][j]-0.05) & (vals[0]<tab['tpeak'][j]+0.05)

    # create the light curve model for the Llamaradas Estelares model
    tm_output = {tm_parameters[x].split('_')[-1]:tab[tm_parameters[x]][j] for
                 x in range(len(tm_parameters))}

    tm_model = fm.llamarades_model(vals[0][q], **tm_output)

    # create the light curve model for the double-peak model
    p_output = {p_parameters[x]:tab[p_parameters[x]][j] for x in range(len(p_parameters))}
    p_model = fm.pietras_double(vals[0][q]-vals[0][q][0], **p_output)

    if i == 0:
        labels=['Double-peak', 'Llamaradas Estelares']
    else:
        labels=['', '']

    # plot the data
    flare_axes[i].plot(vals[0][q]-tab['tpeak'][j], vals[1][q]-1,
                       'o', color='#323031')

    # plot the double-peak model
    flare_axes[i].plot(vals[0][q]-tab['tpeak'][j], p_model,
                       color='#f4a261', lw=2.5, label=labels[0])

    # plot the Llamaradas Estelares model
    flare_axes[i].plot(vals[0][q]-tab['tpeak'][j],
                       tm_model,
                       color='#2a9d8f', lw=2.5, label=labels[1])

    # add the subplot label
    flare_axes[i].text(-0.048, offset[i],
                       s='({})'.format(letter[i]),
                       fontweight='bold', fontsize=16)

    flare_axes[i].set_xlim(-0.05, 0.05)
    flare_axes[i].set_xticks([-0.04, 0.0, 0.04])

    flare_axes[i].set_yticks(np.round(np.linspace(0, np.nanmax(vals[1][q]-1), 3), 1))

    if i < 4:
        flare_axes[i].set_xticklabels([])
    if i == 0:
        flare_axes[i].legend(bbox_to_anchor=(0.22, 1.1, 2.3, .102), loc=3,
                             ncol=2, borderaxespad=0.,
                             fontsize=14)
    flare_axes[i].set_rasterized(True)

flare_axes[-2].set_xlabel('Time from T$_{peak}$ [days]', x=1.07)
flare_axes[2].set_ylabel('Normalized Flux')

plt.savefig(os.path.join(figures_dir,'model_fit.pdf'), dpi=300, bbox_inches='tight')
