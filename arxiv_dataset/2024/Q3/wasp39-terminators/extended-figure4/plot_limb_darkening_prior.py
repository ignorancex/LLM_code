import numpy as np

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from scipy.ndimage import gaussian_filter

import pickle
import glob

import juliet

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['axes.linewidth']=1
fs=15

fig = plt.figure(figsize=[6.69, 3.0])

gs = GridSpec(250, 1, figure=fig, wspace=0.05, hspace=0)
cut = 150
hspace = 10
ax1 = fig.add_subplot(gs[:, 0])

w_model, u1_model, u2_model = np.loadtxt('model_ld.txt', unpack = True)
wavs_100, p1_100, p1_err_100, p2_100, p2_err_100, covariances_100 = np.loadtxt('ld_coeffs_w39_prism100.txt', unpack = True)
wavs_30, p1_30, p1_err_30, p2_30, p2_err_30, covariances_30 = np.loadtxt('ld_coeffs_w39_prism30.txt', unpack = True)

ax1.errorbar(wavs_100, p1_100, p1_err_100, fmt = '.', color = 'purple', alpha = 0.5)
ax1.errorbar(wavs_100, p2_100, p2_err_100, fmt = '.', color = 'royalblue', alpha = 0.5)

ax1.errorbar(wavs_30, p1_30, p1_err_30, fmt = 'o', color = 'purple', ecolor = 'purple', ms = 10, elinewidth = 3, label = '$u_1$ (linear)')
ax1.errorbar(wavs_30, p2_30, p2_err_30, fmt = 'o', color = 'royalblue', ecolor = 'royalblue', ms = 10, elinewidth = 3, label = '$u_2$ (quadratic)')

font = font_manager.FontProperties(style='normal',
                                   size = fs)

leg = ax1.legend(loc = 'upper right', prop = font)
leg.get_frame().set_linewidth(0.0)

plt.plot(w_model, u1_model, color = 'purple', alpha = 0.5, lw = 3 )
plt.plot(w_model, u2_model, color = 'royalblue', alpha = 0.5, lw = 3)

#ax1.text(2.1,12000, s = r'Case 1: True model has no limb assymetries',  weight='bold')

ax1.set_ylim(0.04,0.25)
ax1.set_xlim(2.0,5.3)
ax1.set_ylabel('limb darkening coefficients', fontsize = fs, fontstyle = 'normal')
ax1.set_xlabel('wavelength (um)', fontsize = fs, fontstyle = 'normal')
ax1.tick_params(which = 'both', direction = 'in', labelsize = fs, axis='both', top=True, left=True, right=True, zorder=100)
#ax1.axes.xaxis.set_ticklabels([])
ax1.grid(color = 'grey', linestyle = '--', linewidth = 0.5, alpha = 0.25)

plt.savefig('coefficients_priorlds.pdf', dpi=350, bbox_inches='tight', transparent=True)
