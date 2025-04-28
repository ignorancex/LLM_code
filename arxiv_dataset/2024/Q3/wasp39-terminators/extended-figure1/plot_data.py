import os, sys
import seaborn as sns
import pickle
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.patches import Arrow
from matplotlib.gridspec import GridSpec
from astropy.table import Table, Column, vstack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from utils import (load_plt_params, convolve_model, convolve_model_xy,
                   load_parula, ibin)

# Catwoman, Nestor:
w_ne, ev_ne, everr_ne, mo_ne, moerr_ne = np.loadtxt('catwoman_res100_priorlds.txt', unpack = True, usecols = (0, 1, 2, 3, 4)) 

# Catwoman, Matthew:
w_mm, ev_mm, everr_mm, mo_mm, moerr_mm = np.loadtxt('murphy_catwomanfit_binscale4.txt', unpack = True, usecols = (0, 1, 2, 3, 4)) 

# Half ingress/egress James:
w_jk, ev_jk, everr_jk, mo_jk, moerr_jk = np.loadtxt('Tiberius_res_100.txt', unpack = True, usecols = (0, 7, 8, 5, 6)) 

ev_jk = ev_jk*0.5
everr_jk = everr_jk*0.5
mo_jk = mo_jk*0.5
moerr_jk = moerr_jk*0.5

# Set the matplotlib parameters
pltparams = load_plt_params()
parula = load_parula()
COLOR = pltparams[pltparams['name']=='text.color']['value'][0]

fig = plt.figure(figsize=(10,12))
fig.set_facecolor('w')

#color_palette = sns.color_palette("coolwarm_r",3)
color_palette = sns.color_palette('Set2',3)

gs = GridSpec(450+50, 1, figure=fig)#, height_ratios=[2,2,1])#, wspace=0.05, hspace=0)

text_x = 2.1
text_y = 11700
text_fs = 30
cut = 150 
hspace = 10
ax1 = fig.add_subplot(gs[:150, 0]) 

ax1.set_prop_cycle('color', color_palette)
ax1.errorbar(w_ne, ev_ne, everr_ne, fmt = 'o', label = 'catwoman (NE)')
ax1.errorbar(w_mm, ev_mm, everr_mm, fmt = 's', label = 'catwoman (MM)')
ax1.errorbar(w_jk, ev_jk, everr_jk, fmt = 'd', label = 'Tiberius')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol = 3, frameon=False)

ax1.set_xlim(2,np.max(w_ne)+0.05)
ax1.set_ylim(9500,12300)

ax1.set_ylabel('Evening depth (ppm)')
ax1.text(text_x, text_y,'a', fontsize = text_fs)

ax2 = fig.add_subplot(gs[150+25:300+25,0])

ax2.text(text_x, text_y, 'b', fontsize = text_fs)
ax2.set_prop_cycle('color', color_palette)
ax2.errorbar(w_ne, mo_ne, moerr_ne, fmt = 'o', label = 'catwoman (transitspectroscopy)')
ax2.errorbar(w_mm, mo_mm, moerr_mm, fmt = 's', label = 'catwoman (MM)')
ax2.errorbar(w_jk, mo_jk, moerr_jk, fmt = 'd', label = 'half-egress (Tiberius)')

ax2.set_ylabel('Morning depth (ppm)')
ax2.set_xlim(2,np.max(w_ne)+0.05)
ax2.set_ylim(9500,12300)

ax3 = fig.add_subplot(gs[300+50:450+50,0])

ax3.text(text_x, 25, 'c', fontsize = text_fs)
w_t, dt, dterr = np.loadtxt('time_shift.txt', unpack = True, usecols = (0, 4, 5))

dt = dt - np.nanmedian(dt)
ax3.errorbar(w_t, dt, dterr, fmt = '.', color = 'black')
ax3.set_ylabel('$\Delta\ T_0$ (sec)')
ax3.set_xlabel('Wavelength ($\mu$m)')
ax3.set_xlim(2,np.max(w_ne)+0.05)

# Deactivate ticks for first two plots:
for ax in [ax1, ax2]:

    ax.tick_params(labelbottom=False)

plt.savefig('data.pdf',
            dpi=250,#, rasterize=True,
            #transparent=True,
            bbox_inches='tight')
