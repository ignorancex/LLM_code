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
from analysis_utils import logpdf

set_rcparams()

""" GET THE DATA """
table = Table.read(os.path.join(data_dir,'rossby_stats_v6.csv'), format='csv')
lo = np.load(os.path.join(data_dir,'r0_samples_lo_v6.npy'), allow_pickle=True)
hi = np.load(os.path.join(data_dir,'r0_samples_hi_v6.npy'), allow_pickle=True)
amp_min = 10**-1.8
bins = np.logspace(-2,2,100)
R0 = 0.13617449664429532


logasamp = np.log10(np.logspace(np.log10(amp_min), 1.0))
interval = 100

pdf_samp_lo = np.array([logpdf(logasamp, lo[0][i,0], 10.**lo[0][i,1], amp_min)
                        for i in range(0, lo.shape[0], interval)])
pct_short_lo = np.percentile(pdf_samp_lo, [2.5,5,16,50,84,95,97.5], axis=0)

pdf_samp_hi = np.array([logpdf(logasamp, hi[0][i,0], 10.**hi[0][i,1], amp_min)
                        for i in range(0, hi.shape[0], interval)])

pct_short_hi = np.percentile(pdf_samp_hi, [2.5,5,16,50,84,95,97.5], axis=0)

fig, (ax) = plt.subplots(figsize=(8,3.5), sharex=True,
                              sharey=True)


colors = ['#b75d69', '#1a1423']
alpha, lw = [0.4, 0.2], 2
dat = [lo[1], hi[1]]

""" PLOT THE DATA """
for i in range(len(dat)):
    ax.hist(np.log10(dat[i]), bins=30, color=colors[i], lw=0,
                 alpha=alpha[i], density=True, range=[np.log10(amp_min), 1])
    ax.hist(np.log10(dat[i]), bins=30, fill=None, edgecolor=colors[i],
             lw=lw, density=True, range=[np.log10(amp_min), 1])

""" PLOT THE BEST FIT MODEL """
axes, pct = [ax, ax], [pct_short_lo, pct_short_hi]
for i in range(len(axes)):
    if i == 1:
        labelS = '$R_0 >$' + str(np.round(R0,3)) + '\n'
        label = r'$\alpha = ' + '{0:.3f}'.format(table['alpha'][i]) + '_{' + \
                '-{0:.3f}'.format(table['alpha_lo'][i]) + '}^{' + \
                '+{0:.3f}'.format(table['alpha_hi'][i]) + '}$'
    else:
        labelS = '$R_0 \leq$' + str(np.round(R0,3)) + '\n'
        label = r'$\alpha = ' + '{0:.3f} \pm {1:.3f}$'.format(table['alpha'][i],
                                                            table['alpha_lo'][i])
    labelA = r'$A_* = ' + '{0:.3f}'.format(table['A'][i]) + '_{' + \
                '-{0:.3f}'.format(table['A_lo'][i]) + '}^{' + \
                '+{0:.3f}'.format(table['A_hi'][i]) + '}$'

    axes[i].plot(logasamp, pct[i][3,:], color=colors[i], lw=2,
                 label=labelS + label + '\n' + labelA)
    axes[i].fill_between(logasamp, pct[i][0,:], pct[i][1,:],
                     color=colors[i], alpha=0.15, lw=0, zorder=3)
    axes[i].fill_between(logasamp, pct[i][1,:], pct[i][2,:],
                     color=colors[i], alpha=0.3, lw=0, zorder=3)
    axes[i].fill_between(logasamp, pct[i][2,:], pct[i][4,:],
                     color=colors[i], alpha=0.6, lw=0, zorder=3)
    axes[i].fill_between(logasamp, pct[i][4,:], pct[i][5,:],
                     color=colors[i], alpha=0.3, lw=0, zorder=3)
    axes[i].fill_between(logasamp, pct[i][5,:], pct[i][6,:],
                     color=colors[i], alpha=0.15, lw=0, zorder=3)
    axes[i].set_rasterized(True)

plt.yscale('log')
ax.set_xlabel('log$_{10}$(Amplitude)')
ax.set_ylabel('$dp/dA$')

plt.ylim(10**-3, 4)
plt.xlim(-1.8, 0.5)
leg = plt.legend(ncol=2, fontsize=16, markerscale=4,
                 bbox_to_anchor=(0.016, 1.46), loc=2, borderaxespad=0.)
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)

plt.savefig(os.path.join(figures_dir, 'truncated.pdf'),
            bbox_inches='tight', dpi=300)
