import numpy as np

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import pickle
import glob

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['axes.linewidth']=1
fs=11

wavs_30, thediff_30, thediff_err_up_30, thediff_err_down_30, p1_30, p1_err_30, p2_30, p2_err_30, covariances_30 = np.loadtxt('bin1_eureka_fixedlds_fake30.txt', unpack = True)
wavs_100, thediff_100, thediff_err_up_100, thediff_err_down_100, p1_100, p1_err_100, p2_100, p2_err_100, covariances_100 = np.loadtxt('bin1_eureka_fixedlds_fake100.txt', unpack = True)

#fig = plt.figure(figsize=[6.69, 6.0])
fig = plt.figure(figsize=[4.3, 4.0])

gs = GridSpec(250, 1, figure=fig, wspace=0.05, hspace=0)
cut = 150
hspace = 10
ax1 = fig.add_subplot(gs[:cut, 0])

ax1.errorbar(wavs_100, p1_100, p1_err_100, fmt = '.-', color = 'tomato', alpha = 0.5)
ax1.errorbar(wavs_100, p2_100, p2_err_100, fmt = '.-', color = 'cornflowerblue', alpha = 0.5)

ax1.errorbar(wavs_30, p1_30, p1_err_30, fmt = 'o', color = 'tomato', ecolor = 'tomato', ms = 8, elinewidth = 3, label = 'Evening')
ax1.errorbar(wavs_30, p2_30, p2_err_30, fmt = 'o', color = 'cornflowerblue', ecolor = 'cornflowerblue', ms = 8, elinewidth = 3, label = 'Morning')

font = font_manager.FontProperties(weight='bold',
                                   style='normal',
                                   size = fs)

ax1.text(2.1,11900, s = r'a. true model has no limb asymmetries',  fontsize = 10)

ax1.set_ylim(9500,12300)
ax1.set_xlim(2.0,5.3)
ax1.set_ylabel('limb depth (ppm)', fontsize = fs, fontstyle = 'normal')
ax1.tick_params(which = 'both', direction = 'in', labelsize = fs, axis='both', top=True, left=True, right=True, zorder=100)
ax1.axes.xaxis.set_ticklabels([])
ax1.grid(color = 'grey', linestyle = '--', linewidth = 0.5, alpha = 0.25)
ax1.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax2 = fig.add_subplot(gs[cut+hspace:,0])

ax2.text(2.1,1000, s = r'b. residuals',  fontsize = 10)
ax2.plot([0.6,5.3], [0., 0.], '--', color = 'black')
ax2.errorbar(wavs_100, thediff_100, [thediff_err_down_100, thediff_err_up_100], fmt = '.', alpha = 0.5, color = 'grey')
ax2.errorbar(wavs_30, thediff_30, [thediff_err_down_30, thediff_err_up_30], fmt = 'o', ms = 8, elinewidth = 3, mfc = 'black', mec = 'black', ecolor = 'black')
ax2.set_xlim(2.0,5.3)
ax2.set_ylim(-1500,1500)
ax2.yaxis.set_major_locator(tck.FixedLocator([-1000,-500,0,500,1000]))
ax2.set_ylabel('o-c (ppm)', fontsize = fs, fontstyle = 'normal')
ax2.set_xlabel('wavelength (um)', fontsize = fs, fontstyle = 'normal')
ax2.tick_params(which = 'both', direction = 'in', labelsize = fs, axis='both', top=True, left=True, right=True, zorder=100)
ax2.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



plt.savefig('limb_experiment_diff.pdf', dpi=350, bbox_inches='tight', transparent=True)
