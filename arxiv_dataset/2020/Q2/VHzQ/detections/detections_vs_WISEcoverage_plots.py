'''
WISE detections and colors of Very High redshift quasars
'''
import numpy as np

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
from matplotlib import gridspec

from mpl_toolkits.axes_grid1.axes_divider  import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.colorbar      import colorbar


## READ-IN THE DATA FILE(S)
path = '/cos_pc19a_npr/programs/quasars/highest_z/data/WISE/'
all_VHzQs  = ascii.read(path+'VHzQs463_AllWISE_W1234_coverage.tbl')

## Setting up the variables for easier use
w1mpro = all_VHzQs['w1mpro']
w2mpro = all_VHzQs['w2mpro']
w3mpro = all_VHzQs['w3mpro']
w4mpro = all_VHzQs['w4mpro']
w1snr = all_VHzQs['w1snr']
w2snr = all_VHzQs['w2snr']
w3snr = all_VHzQs['w3snr']
w4snr = all_VHzQs['w4snr']
w1cov = all_VHzQs['w1cov']
w2cov = all_VHzQs['w2cov']
w3cov = all_VHzQs['w3cov']
w4cov = all_VHzQs['w4cov']


print()
print('  w1mpro min, max; {:7.3f} {:7.3f} '.format(w1mpro.min(), w1mpro.max()),  '    w1snr min, max;  {:7.3f} {:7.3f} '.format(w1snr.min(), w1snr.max()))
print('  w2mpro min, max; {:7.3f} {:7.3f} '.format(w2mpro.min(), w2mpro.max()),  '    w2snr min, max;  {:7.3f} {:7.3f} '.format(w2snr.min(), w2snr.max()))
print('  w3mpro min, max; {:7.3f} {:7.3f} '.format(w3mpro.min(), w3mpro.max()),  '    w3snr min, max;  {:7.3f} {:7.3f} '.format(w3snr.min(), w3snr.max()))
print('  w4mpro min, max; {:7.3f} {:7.3f} '.format(w4mpro.min(), w4mpro.max()),  '    w4snr min, max;  {:7.3f} {:7.3f} '.format(w4snr.min(), w4snr.max()))
print()


##
##  Making the plot
##
##  May fave new line ;-=)
#plt.style.use('dark_background')

## 4 plots all in (one) row
#fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14.0, 5.0))
## If 1 by 4:: 
left   = 0.04   # the left side of the subplots of the figure
right  = 0.96   # the right side of the subplots of the figure
bottom = 0.10   # the bottom of the subplots of the figure
top    = 0.90   # the top of the subplots of the figure
wspace = 0.48   # the amount of width reserved for blank space between subplots
hspace = 0.20   # the amount of height reserved for white space between subplots

## 4 plots all 2x2 grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.4, 8.0))
left   = 0.12   # the left side of the subplots of the figure
right  = 0.92   # the right side of the subplots of the figure
bottom = 0.10   # the bottom of the subplots of the figure
top    = 0.94   # the top of the subplots of the figure
wspace = 0.44   # the amount of width reserved for blank space between subplots
hspace = 0.24   # the amount of height reserved for white space between subplots

plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ls       = 'solid'
lw       = 1.0
ms       = 50.
ms_large = ms*3.
fontsize = 14
alpha    = 1.00
## define the colormap
cmap = plt.cm.inferno_r


##
##  w1mpro  vs.  w1snr,   top left
## 
xmin = 13.00; xmax =  19.00; ymin =  -2.00; ymax =  45.00
ax1.scatter(w1mpro, w1snr, s=ms*1.8,  alpha=1.0, color='k')
im1 = ax1.scatter(w1mpro, w1snr, s=ms,  cmap= cmap, alpha=alpha, c=w1cov, vmin=12, vmax=110) 
ax1.set_xlim((xmin, xmax))
ax1.set_ylim((ymin, ymax))
ax1.tick_params('x', direction='in', which='both', bottom='True', top='True', left='True', right='True')
ax1.tick_params('y', direction='in', which='both', bottom='True', top='True', left='True', right='False')
ax1.set_xlabel('W1 Mag', fontsize=fontsize)
ax1.set_ylabel('W1 SNR', fontsize=fontsize)

ax1_divider = make_axes_locatable(ax1)
cax1 = ax1_divider.append_axes("right", size="8%", pad="0%")
cb1 = colorbar(im1, cax=cax1)

##
##  w2mpro  vs.  w2snr,   top right
##
xmin = 13.00; xmax =  19.00; ymin = -2.00; ymax =  45.00    ## `W1' limits
#xmin = 13.00;  xmax =  19.00; ymin = -5.00; ymax = 40.00   ## `W2' limits
ax2.scatter(w2mpro, w2snr, s=ms*2.0,  alpha=1.0, color='k')
im2 = ax2.scatter(w2mpro, w2snr, s=ms,  cmap=cmap, alpha=alpha, label='237 VHzQs', c=w2cov, vmin=12, vmax=110)
ax2.set_xlim((xmin, xmax))
ax2.set_ylim((ymin, ymax))
ax2.tick_params('x', direction='in', which='both', bottom='True', top='True', left='True', right='True')
ax2.tick_params('y', direction='in', which='both', bottom='True', top='True', left='True', right='True')
ax2.set_xlabel('W2 Mag', fontsize=fontsize)
ax2.set_ylabel('W2 SNR', fontsize=fontsize)

ax2_divider = make_axes_locatable(ax2)
cax2 = ax2_divider.append_axes("right", size="8%", pad="0%")
cb2 = colorbar(im2, cax=cax2)
cb2.ax.set_ylabel('number of exposures', rotation='270', fontsize=fontsize, labelpad=10)
cb2.ax.tick_params(labelsize=fontsize) 


##
##  w1mpro  vs.  w1snr,   top left
## 
xmin =   9.85; xmax =  13.80; ymin =  -3.00; ymax =  13.20
ax3.scatter(w3mpro, w3snr, s=ms*2.2,  alpha=1.0, color='k')
im3 = ax3.scatter(w3mpro, w3snr, s=ms,  cmap= cmap,alpha=alpha, label='237 VHzQs', c=w3cov, vmin=6, vmax=60)
ax3.set_xlim((xmin, xmax))
ax3.set_ylim((ymin, ymax))
ax3.tick_params('x', direction='in', which='both', bottom='True', top='True', left='True', right='True')
ax3.tick_params('y', direction='in', which='both', bottom='True', top='True', left='True', right='False')
ax3.set_xlabel('W3 Mag', fontsize=fontsize)
ax3.set_ylabel('W3 SNR', fontsize=fontsize)

ax3_divider = make_axes_locatable(ax3)
cax3 = ax3_divider.append_axes("right", size="8%", pad="0%")
cb3 = colorbar(im3, cax=cax3)

##
##  w4mpro  vs.  w4snr,   top left
## 
xmin =   7.00; xmax =  10.00; ymin =   -4.00; ymax =  9.3
ax4.scatter(w4mpro, w4snr, s=ms*2.4,  alpha=1.0, color='k')
im4 = ax4.scatter(w4mpro, w4snr, s=ms,   cmap= cmap, alpha=alpha, label='2?? VHzQs',  c=w4cov, vmin=6, vmax=60)
ax4.set_xlim((xmin, xmax))
ax4.set_ylim((ymin, ymax))
ax4.tick_params('x', direction='in', which='major', bottom='True', top='True', left='True', right='True')
ax4.tick_params('x', direction='in', which='minor', bottom='True', top='True', left='True', right='True')
ax4.tick_params('y', direction='in', which='major', bottom='True', top='True', left='True', right='False')
ax4.tick_params('y', direction='in', which='minor', bottom='True', top='True', left='True', right='False')
ax4.set_xlabel('W4 Mag', fontsize=fontsize)
ax4.set_ylabel('W4 SNR', fontsize=fontsize)

ax4_divider = make_axes_locatable(ax4)
cax4 = ax4_divider.append_axes("right", size="10%", pad="0%")
cb4 = colorbar(im4, cax=cax4)
cb4.ax.set_ylabel('number of exposures', rotation='270', fontsize=fontsize, labelpad=16)
cb4.ax.tick_params(labelsize=fontsize) 



#plt.show()
plt.savefig('WISEmag_vs_coverage_temp.png', format='png')
plt.savefig('WISEmag_vs_coverage_temp.pdf', format='pdf')
plt.close(fig)






