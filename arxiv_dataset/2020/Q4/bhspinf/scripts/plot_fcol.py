import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from pdf_trans import *
from gs_stats import *

'''
python plot_fcol.py
'''

fout = '../ms/figures/fcol_minmax_spin.eps'

#====================================================================================================
# COLLECT THE RESULTS

dir      = '../data/fr/'
fh5_list = ['fr_LMCX1.h5', 'fr_U1543.h5', 'fr_J1655.h5', 'fr_J1550.h5', 'fr_M33X7.h5', 'fr_LMCX3.h5', 'fr_H1743.h5', 'fr_A0620.h5']
x_lab    = [r"LMC X--1", r"4U 1543", r"GRO J1655", r"XTE J1550", r"M33 X--7", r"LMC X--3", r"H1743", r"A0620"]
markers  = ['D',  'o',  's',  '*',  'X',  'X',  'X',  'X']
s        = 150
msizes   = [s,    s,    s,    2*s,  s,    s,    s,    s  ]

fcolCF, fcol_a0, fcol_a1, fcol_aFe = [], [], [], []
for fh5 in fh5_list:
    f = h5py.File(dir+fh5, 'r')
    fcolCF.append(f.get('fcolCF')[()])
    fcol_a0.append(f.get('fcol_a0')[()])
    fcol_a1.append(f.get('fcol_aMax')[()])
    if (f.get('fcol_aFe') is None): fcol_aFe.append(f.get('fcol_aFe'))
    else:                           fcol_aFe.append(f.get('fcol_aFe')[()])
    f.close()
print("\nSources: ", x_lab)
print("\nfcol CF  = ", fcolCF)
print("\nfcol a0  = ", fcol_a0)
print("\nfcol a1  = ", fcol_a1)
print("\nfcol aFe = ", fcol_aFe)

#====================================================================================================
# PLOT: f_col where a = [0.998, 0, aFe] for each source

fcol_min, fcol_max = 0.5, 3.0

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 5.6
Lgap, Rgap, Bgap, Tgap = 0.15, 0.05, 0.225, 0.05
left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
xbox = 1.0 - (Lgap + Rgap)
ybox = 1.0 - (Bgap + Tgap)
fs, fs_sm, fs_vsm, lw, pad, tlmaj, tlmin = 28, 24, 20, 2, 10, 10, 5
xlab_pos = (0.5, -0.125)
ylab_pos = (-0.15, 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.5

# Axes limits and tick increments
x_min, x_max = 0, 1
y_min, y_max = fcol_min, fcol_max
yinc_maj, yinc_min = 0.5, 0.1
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.set_ylabel(r"$f_{\rm col}$", fontsize=fs, ha='center', va='center', rotation=0, labelpad=pad)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('y', left=True, right=True,  which='major', direction='out', length=tlmaj, width=lw, labelsize=fs_sm, pad=pad)
ax.tick_params('y', left=True, right=True,  which='minor', direction='out', length=tlmin, width=lw)
ax.tick_params('x', top=False, bottom=False, which='both',  labelsize=fs_vsm, pad=pad)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# Align the axes labels (using the axes coordinate system)
#ax.xaxis.set_label_coords(xlab_pos[0], xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], ylab_pos[1], transform=ax.transAxes)#fig.transFigure

# x-axis labels
#x_formatter = FixedFormatter([r"GRO J1655--40", r"", r"", r"", r"", r"", r"", r"", r"", r""])
#x_locator   = FixedLocator([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
#ax.xaxis.set_major_formatter(x_formatter)
#ax.xaxis.set_major_locator(x_locator)
#x_loc = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
#x_lab = [r"GRS 1915", r"Cyg X--1", r"LMC X--1", r"4U 1543", r"GRO J1655", r"XTE J1550", r"M33 X--7", r"LMC X--3", r"H1743", r"A0620"]
Nsrc   = len(x_lab)
dxHalf = 0.5 / Nsrc
x_loc  = np.linspace(dxHalf, 1.0-dxHalf, Nsrc)
plt.xticks(x_loc, x_lab, rotation=45)

# Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

# Colormap
fcol_min, fcol_max, Ncb = 0.9, 2.5, 9
cmap = cm.get_cmap('viridis_r')
fcol_edges = np.linspace(fcol_min, fcol_max, Ncb)
fcol_cents = 0.5 * (fcol_edges[:-1] + fcol_edges[1:])
fcol_norm  = BoundaryNorm(fcol_edges, cmap.N)
s_map      = cm.ScalarMappable(norm=fcol_norm, cmap=cmap)

# Plot
for i in np.arange(Nsrc):
    # fcolCF
    color_CF = cmap(fcol_norm(fcolCF[i]))
    ax.scatter([x_loc[i],x_loc[i]], [fcolCF[i],fcolCF[i]], marker='.', s=100, facecolors='r', edgecolors=None, linewidths=lw, zorder=3)
    # fcol: a = 0
    ax.scatter([x_loc[i],x_loc[i]], [fcol_a0[i],fcol_a0[i]], marker=markers[i], s=msizes[i], facecolors='w', edgecolors='k', linewidths=lw, zorder=3)
    ax.plot([x_loc[i],x_loc[i]], [y_min,fcol_a0[i]], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    if (i <  0.5*Nsrc): ax.plot([x_min,x_loc[i]], [fcol_a0[i],fcol_a0[i]], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    if (i >= 0.5*Nsrc): ax.plot([x_loc[i],x_max], [fcol_a0[i],fcol_a0[i]], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    # fcol: a = 0.998
    ax.scatter([x_loc[i],x_loc[i]], [fcol_a1[i],fcol_a1[i]], marker=markers[i], s=msizes[i], facecolors='dimgray', edgecolors='k', linewidths=lw, zorder=3)
    ax.plot([x_loc[i],x_loc[i]], [y_min,fcol_a1[i]], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    if (i <  0.5*Nsrc): ax.plot([x_min,x_loc[i]], [fcol_a1[i],fcol_a1[i]], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    if (i >= 0.5*Nsrc): ax.plot([x_loc[i],x_max], [fcol_a1[i],fcol_a1[i]], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    # fcol: aCF = aFe
    if (fcol_aFe[i] is not None):
        color_aFe = cmap(fcol_norm(fcol_aFe[i]))
        #ax.scatter([x_loc[i],x_loc[i]], [fcol_aFe[i],fcol_aFe[i]], marker='_', s=s, facecolors=color_aFe, edgecolors=color_aFe, linewidths=lw, zorder=3)
        ax.scatter([x_loc[i],x_loc[i]], [fcol_aFe[i],fcol_aFe[i]], marker='_', s=s, facecolors='b', edgecolors='b', linewidths=lw, zorder=3)

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()
