import numpy as np
import pylab as plt
from matplotlib.ticker import MultipleLocator

'''
python plot_aFe_aCF.py
'''

# Output file name
outdir = '../ms/figures/'
fout   = outdir + 'aFe_aCF.eps'

# USE z-SCORE TO CONVERT TO 68% UNCERTAINTY

# Continuum fitting measured black hole spins and errors: [value, -err, +err]
aCF_U1543 = [0.8, 0.1, 0.1]    # 68%...?
aCF_CygX1 = [0.9985, 0.0148, 0.0005] # 68%
aCF_1915  = [0.98]             # 3-sigma
aCF_J1655 = [0.7, 0.1, 0.1]    # 68%...?
aCF_LMCX1 = [0.92, 0.07, 0.05] # 68%
aCF_J1550 = [0.34, 0.28, 0.20] # 68%

# Iron line measured black hole spins and errors: [value, -err, +err]
aFe_U1543 = [0.3, 0.1, 0.1]     # 68%
aFe_CygX1 = [0.97, 0.02, 0.014] # 90%
aFe_1915  = [0.98, 0.01, 0.01]  # 68%
aFe_J1655 = [0.92, 0.02, 0.02]  # 90%
aFe_LMCX1 = [0.97, 0.25, 0.02]  # 68%
aFe_J1550 = [0.55, 0.22, 0.15]  # 68%

# Use z-scores to convert uncertainties to 1-sigma level
z90 = 1.645
aFe_CygX1[1], aFe_CygX1[2] = aFe_CygX1[1]/z90, aFe_CygX1[2]/z90
aFe_J1655[1], aFe_J1655[2] = aFe_J1655[1]/z90, aFe_J1655[2]/z90

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 8.4
left, right, bottom, top = 0.15, 0.97, 0.15, 0.97
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 12, 10, 5

# Plotting preferences
ts          = 16.0  # Text size
legfs       = 16.0  # Legend font size
symsize     = 300   # Symbol size
symsize_leg = 150   # Symbol size (for legend)
csize       = 0.0   # Capsize for error bars
csize_arrow = 5.0
elw         = 2.0   # Width of error bars
mew         = 2.0   # Marker edge width
limlen      = 0.1   # Length of x error upper/lower limits arrow

a_min, a_max = -1,  1
xmin,  xmax  = 0.0, 1.1
ymin,  ymax  = 0.0, 1.1
xinc_maj, xinc_min = 0.2, 0.05
yinc_maj, yinc_min = 0.2, 0.05
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

mark_U1543 = 'o'  # circle
mark_CygX1 = '^'  # up-triangle
mark_1915  = 'v'  # down-triangle
mark_J1655 = 's'  # square
mark_LMCX1 = 'D'  # diamond
mark_J1550 = '*'  # star

fig = plt.figure(figsize=(8.4, 8.4), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.axhline(y=ymin, xmin=xmin, xmax=xmax, linewidth=lw, color='k')
ax.axhline(y=ymax, xmin=xmin, xmax=xmax, linewidth=lw, color='k')
ax.axvline(x=xmin, ymin=ymin, ymax=ymax, linewidth=lw, color='k')
ax.axvline(x=xmax, ymin=ymin, ymax=ymax, linewidth=lw, color='k')
ax.set_xlabel(r"${\rm Iron\ Line\ Black\ Hole\ Spin,}\ a^{\rm Fe}$",         fontsize=fs, labelpad=pad)
ax.set_ylabel(r"${\rm Continuum\ Fitting\ Black\ Hole\ Spin,}\ a^{\rm CF}$", fontsize=fs, labelpad=pad)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.tick_params('both', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='in', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.plot([xmin,xmax], [ymin,ymax], 'k--', linewidth=lw)

s_U1543 = ax.scatter(aFe_U1543[0], aCF_U1543[0], marker=mark_U1543, s=symsize, facecolors='w', edgecolors='k', linewidths=elw, zorder=3)
ax.errorbar([aFe_U1543[0],aFe_U1543[0]], [aCF_U1543[0],aCF_U1543[0]], \
            xerr=[[aFe_U1543[1],aFe_U1543[1]], [aFe_U1543[2],aFe_U1543[2]]], \
            yerr=[[aCF_U1543[1],aCF_U1543[1]], [aCF_U1543[2],aCF_U1543[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)

s_CygX1 = ax.scatter(aFe_CygX1[0], aCF_CygX1[0], marker=mark_CygX1, s=symsize, facecolors='w', edgecolors='k', linewidths=elw, zorder=3)
ax.errorbar([aFe_CygX1[0],aFe_CygX1[0]], [aCF_CygX1[0],aCF_CygX1[0]], \
            yerr=[[0.0,0.0], [1.0-aCF_CygX1[0],1.0-aCF_CygX1[0]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)
ax.errorbar([aFe_CygX1[0],aFe_CygX1[0]], [aCF_CygX1[0],aCF_CygX1[0]], \
            xerr=[[aFe_CygX1[1],aFe_CygX1[1]], [aFe_CygX1[2],aFe_CygX1[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)

s_1915  = ax.scatter(aFe_1915[0], aCF_1915[0], marker=mark_1915, s=symsize, facecolors='w', edgecolors='k', linewidths=elw, zorder=3)
ax.errorbar([aFe_1915[0],aFe_1915[0]], [aCF_1915[0],aCF_1915[0]], \
            yerr=[[0.0,0.0], [1.0-aCF_1915[0],1.0-aCF_1915[0]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize_arrow, elinewidth=elw, color='k', ecolor='k', zorder=2, ylolims=True)
ax.errorbar([aFe_1915[0],aFe_1915[0]], [aCF_1915[0],aCF_1915[0]], \
            xerr=[[aFe_1915[1],aFe_1915[1]], [aFe_1915[2],aFe_1915[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)

s_J1655 = ax.scatter(aFe_J1655[0], aCF_J1655[0], marker=mark_J1655, s=symsize, facecolors='w', edgecolors='k', linewidths=elw, zorder=3)
ax.errorbar([aFe_J1655[0],aFe_J1655[0]], [aCF_J1655[0],aCF_J1655[0]], \
            yerr=[[aCF_J1655[1],aCF_J1655[1]], [aCF_J1655[2],aCF_J1655[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)
ax.errorbar([aFe_J1655[0],aFe_J1655[0]], [aCF_J1655[0],aCF_J1655[0]], \
            xerr=[[aFe_J1655[1],aFe_J1655[1]], [aFe_J1655[2],aFe_J1655[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)

s_LMCX1 = ax.scatter(aFe_LMCX1[0], aCF_LMCX1[0], marker=mark_LMCX1, s=symsize, facecolors='w', edgecolors='k', linewidths=elw, zorder=3)
ax.errorbar([aFe_LMCX1[0],aFe_LMCX1[0]], [aCF_LMCX1[0],aCF_LMCX1[0]], \
            yerr=[[aCF_LMCX1[1],aCF_LMCX1[1]], [aCF_LMCX1[2],aCF_LMCX1[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)
ax.errorbar([aFe_LMCX1[0],aFe_LMCX1[0]], [aCF_LMCX1[0],aCF_LMCX1[0]], \
            xerr=[[aFe_LMCX1[1],aFe_LMCX1[1]], [aFe_LMCX1[2],aFe_LMCX1[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)

s_J1550 = ax.scatter(aFe_J1550[0], aCF_J1550[0], marker=mark_J1550, s=symsize*2.0, facecolors='w', edgecolors='k', linewidths=elw, zorder=3)
ax.errorbar([aFe_J1550[0],aFe_J1550[0]], [aCF_J1550[0],aCF_J1550[0]], \
            xerr=[[aFe_J1550[1],aFe_J1550[1]], [aFe_J1550[2],aFe_J1550[2]]], \
            yerr=[[aCF_J1550[1],aCF_J1550[1]], [aCF_J1550[2],aCF_J1550[2]]], \
            fmt='none', marker=None, markeredgewidth=mew, capsize=csize, elinewidth=elw, color='k', ecolor='k', zorder=2)

ax.text(aFe_U1543[0], aCF_U1543[0]-0.13, r'$4{\rm U}\ 1543$'+'--'r'$47$', verticalalignment='center', horizontalalignment='center', size=ts)
ax.text(aFe_CygX1[0]-0.06, aCF_CygX1[0], r'${\rm Cyg}$'+'\n'+r'${\rm X}$'+'--'r'$1$', verticalalignment='center', horizontalalignment='center', size=ts)
ax.text(aFe_1915[0]+0.095, aCF_1915[0], r'${\rm GRS}$'+'\n'+r'$1915$', verticalalignment='center', horizontalalignment='right', size=ts)
ax.text(aFe_J1655[0], aCF_J1655[0]-0.13, r'${\rm GRO\ J}1655$'+'--'r'$40$', verticalalignment='center', horizontalalignment='center', size=ts)
ax.text(aFe_LMCX1[0]+0.065, aCF_LMCX1[0]-0.05, r'${\rm LMC}$'+'\n'+r'${\rm X}$'+'--'r'$1$', verticalalignment='center', horizontalalignment='center', size=ts)
ax.text(aFe_J1550[0]+aFe_J1550[2], aCF_J1550[0]-0.05, r'${\rm XTE\ J}1550$'+'--'r'$564$', verticalalignment='center', horizontalalignment='center', size=ts)

fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()
