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
fs=10

#fig = plt.figure(figsize=[6.69, 6.0])
fig = plt.figure(figsize=[4.3, 4.0])

hspace = 10
delta = 50

all_experiments = ['bin1_eureka_fixedlds_fake_badtiming', 
                   'bin1_eureka_fixedlds_fake_badlds_0.01',
                   'bin1_eureka_fixedlds_fake_badecc_omega10']

all_names = [r'c. wrong timing ($3\sigma_{T_0}$)', 
             r'd. wrong limb-darkening ($\Delta u_i = 0.01$)',#r'Case 3: wrong eccentricity ($3\sigma_{e}$)', 
             r'e. wrong eccentricity ($3\sigma_{e}$)']

gs = GridSpec(len(all_experiments)*(delta + hspace) - hspace, 1, figure=fig, wspace=0.05, hspace=0)

axs = []
for i in range(len(all_experiments)):


    wavs_30, thediff_30, thediff_err_up_30, thediff_err_down_30, p1_30, p1_err_30, p2_30, p2_err_30, covariances_30 = np.loadtxt(all_experiments[i]+'_30.txt', unpack = True)
    wavs_100, thediff_100, thediff_err_up_100, thediff_err_down_100, p1_100, p1_err_100, p2_100, p2_err_100, covariances_100 = np.loadtxt(all_experiments[i]+'_100.txt', unpack = True)

    axs.append(fig.add_subplot( gs[i*(delta + hspace):i*(delta + hspace) + delta, 0]) )

    ax = axs[-1]

    ax.text(2.1,800, s = all_names[i])

    ax.plot([0.6,5.3], [0., 0.], '--', color = 'black')
    ax.errorbar(wavs_100, thediff_100, [thediff_err_down_100, thediff_err_up_100], fmt = '.', alpha = 0.5, color = 'grey')
    ax.errorbar(wavs_30, thediff_30, [thediff_err_down_30, thediff_err_up_30], fmt = 'o', ms = 10, elinewidth = 3, mfc = 'black', mec = 'black', ecolor = 'black')
    ax.set_xlim(2.0,5.3)
    ax.set_ylim(-1500,1500)
    ax.yaxis.set_major_locator(tck.FixedLocator([-1000,-500,0,500,1000]))
    ax.set_ylabel('o-c (ppm)', fontsize = fs, fontstyle = 'normal')
    ax.tick_params(which = 'both', direction = 'in', labelsize = fs, axis='both', top=True, left=True, right=True, zorder=100)
    
    if i != 2:

        ax.axes.xaxis.set_ticklabels([])

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
ax.set_xlabel('wavelength (um)', fontsize = fs, fontstyle = 'normal')
ax.tick_params(which = 'both', direction = 'in', labelsize = fs, axis='both', top=True, left=True, right=True, zorder=100)
plt.savefig('limb_all_exps.pdf', dpi=350, bbox_inches='tight', transparent=True)
