import numpy as np
import pylab as plt
import h5py
from NCcmap import *
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator, NullFormatter, FixedLocator, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, SymLogNorm
from scipy.interpolate import RectBivariateSpline
from pdf_trans import spin2risco, risco2spin
from scipy import ndimage

'''
Purpose:
--------
Plot the grid of disk flux correction factors gGR(r,i) needed to go from a standard disk spectrum (Page & Thorne 1974) to kerrbb (Li et al. 2005), which includes GR effects on the photons.

Notes:
------
The axes are flipped!
--> gGR(r,i) is supplied, but in the plot i=x-axis, r=y-axis
'''

dout = '../ms/figures/'

# Collect the grids: gGR(r,i), gNT(r)
fh5      = '../data/GR/gGR_gNT_J1655.h5'
f        = h5py.File(fh5, 'r')
a_grid   = f.get('a_grid')[:]  # [-]
r_grid   = f.get('r_grid')[:]  # [Rg]
i_grid   = f.get('i_grid')[:]  # [deg]
gGR_grid = f.get('gGR')[:,:]   # gGR(r[Rg], i[deg])
gNT_grid = f.get('gNT')[:]     # gNT(r[Rg])
f.close()

# Calculate the partial derivatives
dr = r_grid[1] - r_grid[0]
di = i_grid[1] - i_grid[0]
dgGR_dr_grid = np.gradient(gGR_grid, *[dr, di], edge_order=2)[0]
dgGR_di_grid = np.gradient(gGR_grid, *[dr, di], edge_order=2)[1]
dgNT_dr_grid = np.gradient(gNT_grid, *[dr],     edge_order=2)

# Gaussian smoothing kernels
r_GR,  i_GR  = 0,  0  # gGR(r,i)
r_dGR, i_dGR = 25, 0  # d/dr[gGR(r,i)]
r_NT         = 0      # gNT(r)
r_dNT        = 0      # d/dr[gNT(r)]

# Smooth gGR, d/dr[gGR], gNT, d/dr[gNT] using a Gaussian kernel
gGR     = ndimage.filters.gaussian_filter(gGR_grid,     sigma=[r_GR, i_GR],  mode='nearest')
dgGR_dr = ndimage.filters.gaussian_filter(dgGR_dr_grid, sigma=[r_dGR,i_dGR], mode='nearest')
dgGR_di = ndimage.filters.gaussian_filter(dgGR_di_grid, sigma=[r_dGR,i_dGR], mode='nearest')
gNT     = ndimage.filters.gaussian_filter(gNT_grid,     sigma=r_NT,          mode='nearest')
dgNT_dr = ndimage.filters.gaussian_filter(dgNT_dr_grid, sigma=r_dNT,         mode='nearest')

# Append gGR = N/A for inc=(85-90] and rin=[1,1.1)
r_min    = 1.0
Nr       = len(r_grid)
Nr_full  = int((np.min(r_grid) - r_min) / dr + Nr)
i_max    = 90.0
Ni       = len(i_grid)
Ni_full  = int((i_max - np.max(i_grid)) / di + Ni)
gGR_full = np.zeros([Nr_full, Ni_full]) - 1.0
gGR_full = np.full(shape=[Nr_full, Ni_full], fill_value=-1, dtype=float)
gGR_full[(Nr_full-Nr):Nr_full,0:Ni] = gGR
dgGR_dr_full = np.full(shape=[Nr_full, Ni_full], fill_value=-100, dtype=float)
dgGR_dr_full[(Nr_full-Nr):Nr_full,0:Ni] = dgGR_dr

# r^2 gGR
r2    = (r_grid**2.0)[:,np.newaxis]
r2gGR = r2 * gGR
r2gGR_full = np.zeros([Nr_full, Ni_full]) - 1.0
r2gGR_full[(Nr_full-Nr):Nr_full,0:Ni] = r2gGR

# For a non-spinning black hole, find the disk inclination angle for which gGR = 1
ia0 = np.argmin(np.abs(a_grid - 0.0))
ig1 = np.argmin(np.abs(gGR[ia0,:] - 1.0))
print("")
print("g(a=0, inc) = 1 for inc [deg] = ", np.interp(1.0, gGR[ia0,:], i_grid))

# Transpose the 2D arrays for plotting purposes
gGR          = gGR.transpose()
gGR_full     = gGR_full.transpose()
gGR_inv      = 1.0 / gGR
gGR_inv_full = 1.0 / gGR_full
dgGR_dr      = dgGR_dr.transpose()
dgGR_dr_full = dgGR_dr_full.transpose()
r2gGR        = r2gGR.transpose()
r2gGR_full   = r2gGR_full.transpose()

extent = [np.min(r_grid), np.max(r_grid), np.min(i_grid), np.max(i_grid)]

#====================================================================================================
# PLOT gGR(r,i)

fout = dout + 'gGR.eps'
gGR_min = 0.0
gGR_max = 7.0
print("\nMin/Max gGR = ", np.min(gGR), np.max(gGR))

r_min, r_max = 1.0, 9.0
i_min, i_max = 0.0, 90.0

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 6.72#6.3#5.6
Lgap, Bgap, Tgap = 0.15, 0.175, 0.075
Rgap = Tgap + Bgap - Lgap
xbox = (1.0 - (Lgap + Rgap)) * (ysize/xsize)  # Square
ybox =  1.0 - (Bgap + Tgap)                   # Square
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
xlab_pos = (0.5, -0.15)
ylab_pos = (-0.15, 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.75

# Axes limits and tick increments
x_min, x_max = r_min, r_max
y_min, y_max = i_min, i_max
xinc_maj, xinc_min = 1, 0.5
yinc_maj, yinc_min = 10, 5
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

# Colormap, Bounds, and Normalization
cmap = neut_cmap(s=0.5, n=256)
#cmap = cm.get_cmap('viridis_r')
cmap.set_under('w')
yx_min, yx_max, Ncb = gGR_min, gGR_max, 256
yx_edges = np.linspace(yx_min, yx_max, Ncb)
yx_cents = 0.5 * (yx_edges[:-1] + yx_edges[1:])
yx_norm  = BoundaryNorm(yx_edges, cmap.N)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = plt.axes([Lgap, Bgap, xbox, ybox])
ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
ax.set_ylabel(r"${\rm Inner\ Disc\ Inclination,}\ i_{\rm disc}\ (^{\circ})$", fontsize=fs, ha='center', va='bottom')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('both', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='out', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
#ax.xaxis.set_label_position('top')

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(0.5, xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], 0.5, transform=ax.transAxes)#fig.transFigure

# Contour plot <--- might have to only draw for region that exists
plt.rcParams['contour.negative_linestyle'] = 'solid'
norm = cm.colors.Normalize(vmax=1.0, vmin=-1.0)
levels = [0.75, 1.0, 1.5]
cs = plt.contour(gGR, levels, colors='k', origin='lower', extent=extent, norm=norm, linewidths=lw)
ax.text(8.5, 77, r'$g_{\rm GR} = 1.5$',  rotation=5,   fontsize=fs_sm, ha='right', va='center')
ax.text(8.5, 44, r'$g_{\rm GR} = 1$',    rotation=0,   fontsize=fs_sm, ha='right', va='center')
ax.text(3.5, 25, r'$g_{\rm GR} = 0.75$', rotation=-45, fontsize=fs_sm, ha='right', va='center')

# Image plot
im = plt.imshow(gGR_full, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap=cmap, \
                vmin=gGR_min, vmax=gGR_max, interpolation='nearest', aspect='auto')
#r_edges = np.linspace(r_min, r_max, Nr_full)
#i_edges = np.linspace(i_max, i_max, Ni_full)
#x_mesh, y_mesh = np.meshgrid(r_edges, i_edges)
#im = ax.pcolormesh(x_mesh, y_mesh, gGR_full, cmap=cmap, norm=yx_norm)  # <-- NOT WORKING?!

# Colorbar
cmaj  = 50
cbticks = [0, 1, 2, 3, 4, 5, 6, 7]  #yx_cents[::cmaj]
cbticklabels = [r"$0$", r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$"]
CBgap = 0.033
Lcb = Lgap + xbox + CBgap
Bcb = Bgap
Wcb = 0.05
Hcb = 1.0 - (Bgap + Tgap)
cbar_ax = fig.add_axes([Lcb, Bcb, Wcb, Hcb])
cb = plt.colorbar(im, cax=cbar_ax, cmap=cmap, norm=yx_norm, ticks=cbticks, boundaries=yx_edges, orientation='vertical')#, spacing='proportional', format='%2.2g')
cbar_ax.tick_params('y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
cbar_ax.tick_params('y', direction='out', length=tlmin, width=lw, which='minor')
cbar_ax.set_ylabel(r"$g_{\rm GR}$", fontsize=fs, ha='center', va='center', rotation=0)
#cbar_ax.yaxis.set_label_position('right')
cbar_ax.yaxis.set_label_coords(3.0, 0.5, transform=cbar_ax.transAxes)#fig.transFigure
cbar_ax.yaxis.tick_right()
cbar_ax.minorticks_off()
cbar_ax.set_yticklabels(cbticklabels, minor=False)

# Plot the black hole spin values on the top axis
Nticks_a = 5
a_tick_values = np.linspace(1, -1, Nticks_a)  # a values from -1 --> 1
a_tick_locations = np.zeros(Nticks_a)
for i in np.arange(Nticks_a):
    a_tick_locations[i] = (spin2risco(a=a_tick_values[i]) - x_min) / (x_max - x_min)  # rin values
a_tick_labels = [r"$1$", r"$0.5$", r"$0$", r"$-0.5$", r"$-1$"]
ax_top = ax.twiny()
ax_top.tick_params(axis='x', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tick_locations)
ax_top.set_xticklabels(a_tick_labels)
a_tickmin_locations = [(spin2risco(a=0.9) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.9) - x_min) / (x_max - x_min)]
ax_top.tick_params(axis='x', direction='in', length=tlmin, width=lw, which='minor', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tickmin_locations, minor=True)
ax_top.set_xticklabels(['0.9','','','','','','','','','','','','','','',''], minor=True)
ax_top.set_xlabel(r"$\leftarrow a$", fontsize=fs, ha='left', va='top', labelpad=0, x=1.0+CBgap+Wcb)

# Do this here to avoid the code snippet above messing with the y-axis for some reason
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()

#====================================================================================================
# PLOT d/dr[gGR(r,i)]

fout = dout + 'dgGR_dr.eps'
dgGR_dr_min = -3.0
dgGR_dr_max = 1.5
print("\nMin/Max d/dr[gGR] = ", np.min(dgGR_dr), np.max(dgGR_dr))

r_min, r_max = 1.0, 9.0
i_min, i_max = 0.0, 90.0

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 6.72#6.3#5.6
Lgap, Bgap, Tgap = 0.15, 0.175, 0.075
Rgap = Tgap + Bgap - Lgap
xbox = (1.0 - (Lgap + Rgap)) * (ysize/xsize)  # Square
ybox =  1.0 - (Bgap + Tgap)                   # Square
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
xlab_pos = (0.5, -0.15)
ylab_pos = (-0.15, 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.75

# Axes limits and tick increments
x_min, x_max = r_min, r_max
y_min, y_max = i_min, i_max
xinc_maj, xinc_min = 1, 0.5
yinc_maj, yinc_min = 10, 5
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

# Colormap, Bounds, and Normalization
cmap = neut_cmap(s=0.5, n=256)
#cmap = cm.get_cmap('viridis_r')
cmap.set_under('w')
yx_min, yx_max, Ncb = dgGR_dr_min, dgGR_dr_max, 256
yx_edges = np.linspace(yx_min, yx_max, Ncb)
yx_cents = 0.5 * (yx_edges[:-1] + yx_edges[1:])
yx_norm  = BoundaryNorm(yx_edges, cmap.N)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = plt.axes([Lgap, Bgap, xbox, ybox])
ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
ax.set_ylabel(r"${\rm Inner\ Disc\ Inclination,}\ i_{\rm disc}\ (^{\circ})$", fontsize=fs, ha='center', va='bottom')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('both', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='out', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
#ax.xaxis.set_label_position('top')

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(0.5, xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], 0.5, transform=ax.transAxes)#fig.transFigure

# Contour plot <--- might have to only draw for region that exists
plt.rcParams['contour.negative_linestyle'] = 'solid'
norm = cm.colors.Normalize(vmax=1.0, vmin=-1.0)
levels = [-0.1, 0.0, 0.1]
cs = plt.contour(dgGR_dr, levels, colors='k', origin='lower', extent=extent, norm=norm, linewidths=lw)
ax.text(7.0, 72, r'$\partial_{r} g_{\rm GR} = -0.1$', rotation=20,  fontsize=fs_sm, ha='right', va='center')
ax.text(8.5, 44, r'$\partial_{r} g_{\rm GR} = 0$',    rotation=0,   fontsize=fs_sm, ha='right', va='center')
ax.text(2.5, 22, r'$\partial_{r} g_{\rm GR} = 0.1$',  rotation=-80, fontsize=fs_sm, ha='right', va='center')

# Image plot
im = plt.imshow(dgGR_dr_full, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap=cmap, \
                vmin=dgGR_dr_min, vmax=dgGR_dr_max, interpolation='nearest', aspect='auto')
#r_edges = np.linspace(r_min, r_max, Nr_full)
#i_edges = np.linspace(i_max, i_max, Ni_full)
#x_mesh, y_mesh = np.meshgrid(r_edges, i_edges)
#im = ax.pcolormesh(x_mesh, y_mesh, dgGR_dr_full, cmap=cmap, norm=yx_norm)  # <-- NOT WORKING?!

# Colorbar
cmaj  = 50
cbticks = [-3, -2, -1, 0, 1]  #yx_cents[::cmaj]
cbticklabels = [r"$-3$", r"$-2$", r"$-1$", r"$0$", r"$1$"]  #yx_cents[::cmaj]
CBgap = 0.033
Lcb = Lgap + xbox + CBgap
Bcb = Bgap
Wcb = 0.05
Hcb = 1.0 - (Bgap + Tgap)
cbar_ax = fig.add_axes([Lcb, Bcb, Wcb, Hcb])
cb = plt.colorbar(im, cax=cbar_ax, cmap=cmap, norm=yx_norm, ticks=cbticks, boundaries=yx_edges, orientation='vertical')#, spacing='proportional', format='%2.2g')
cbar_ax.tick_params('y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=0.5*pad)
cbar_ax.tick_params('y', direction='out', length=tlmin, width=lw, which='minor')
cbar_ax.set_ylabel(r"$\frac{\partial g_{\rm GR}}{\partial r_{\rm in}}$", fontsize=fs*1.25, ha='center', va='center', rotation=0)
#cbar_ax.yaxis.set_label_position('right')
cbar_ax.yaxis.set_label_coords(3.4, 0.5, transform=cbar_ax.transAxes)#fig.transFigure
cbar_ax.yaxis.tick_right()
cbar_ax.minorticks_off()
cbar_ax.set_yticklabels(cbticklabels, minor=False)

# Plot the black hole spin values on the top axis
Nticks_a = 5
a_tick_values = np.linspace(1, -1, Nticks_a)  # a values from -1 --> 1
a_tick_locations = np.zeros(Nticks_a)
for i in np.arange(Nticks_a):
    a_tick_locations[i] = (spin2risco(a=a_tick_values[i]) - x_min) / (x_max - x_min)  # rin values
a_tick_labels = [r"$1$", r"$0.5$", r"$0$", r"$-0.5$", r"$-1$"]
ax_top = ax.twiny()
ax_top.tick_params(axis='x', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tick_locations)
ax_top.set_xticklabels(a_tick_labels)
a_tickmin_locations = [(spin2risco(a=0.9) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.9) - x_min) / (x_max - x_min)]
ax_top.tick_params(axis='x', direction='in', length=tlmin, width=lw, which='minor', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tickmin_locations, minor=True)
ax_top.set_xticklabels(['0.9','','','','','','','','','','','','','','',''], minor=True)
ax_top.set_xlabel(r"$\leftarrow a$", fontsize=fs, ha='left', va='top', labelpad=0, x=1.0+CBgap+Wcb)

# Do this here to avoid the code snippet above messing with the y-axis for some reason
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()

#====================================================================================================
# PLOT r^2 gGR(r, i)^2

fout = dout + 'r2gGR.eps'
r2gGR_min = 0.0
r2gGR_max = 125.0
print("")
print("Min/Max (r^2 gGR) = ", np.min(r2gGR), np.max(r2gGR))

r_min, r_max = 1.0, 9.0
i_min, i_max = 0.0, 90.0

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 6.72#6.3#5.6
Lgap, Bgap, Tgap = 0.15, 0.175, 0.075
Rgap = Tgap + Bgap - Lgap
xbox = (1.0 - (Lgap + Rgap)) * (ysize/xsize)  # Square
ybox =  1.0 - (Bgap + Tgap)                   # Square
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
xlab_pos = (0.5, -0.15)
ylab_pos = (-0.15, 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.75

# Axes limits and tick increments
x_min, x_max = r_min, r_max
y_min, y_max = i_min, i_max
xinc_maj, xinc_min = 1, 0.5
yinc_maj, yinc_min = 10, 5
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

# Colormap, Bounds, and Normalization
cmap = neut_cmap(s=0.5, n=256)
#cmap = cm.get_cmap('viridis_r')
cmap.set_under('w')
yx_min, yx_max, Ncb = r2gGR_min, r2gGR_max, 256
yx_edges = np.linspace(yx_min, yx_max, Ncb)
yx_cents = 0.5 * (yx_edges[:-1] + yx_edges[1:])
yx_norm  = BoundaryNorm(yx_edges, cmap.N)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = plt.axes([Lgap, Bgap, xbox, ybox])
ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
ax.set_ylabel(r"${\rm Inner\ Disc\ Inclination,}\ i_{\rm disc}\ (^{\circ})$", fontsize=fs, ha='center', va='bottom')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('both', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='out', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
#ax.xaxis.set_label_position('top')

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(0.5, xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], 0.5, transform=ax.transAxes)#fig.transFigure

# Contour plot <--- might have to only draw for region that exists
plt.rcParams['contour.negative_linestyle'] = 'solid'
norm = cm.colors.Normalize(vmax=1.0, vmin=-1.0)
levels = [5, 25, 50, 100]
cs = plt.contour(r2gGR, levels, colors='k', origin='lower', extent=extent, norm=norm, linewidths=lw)
ax.text(3, 18, r'$r_{\rm in}^{2} g_{\rm GR} = 5$',   rotation=-45,  fontsize=fs_sm, ha='center', va='center')
ax.text(3, 38, r'$r_{\rm in}^{2} g_{\rm GR} = 25$',  rotation=-45,  fontsize=fs_sm, ha='center', va='center')
ax.text(3, 58, r'$r_{\rm in}^{2} g_{\rm GR} = 50$',  rotation=-7.5, fontsize=fs_sm, ha='center', va='center')
ax.text(3, 77, r'$r_{\rm in}^{2} g_{\rm GR} = 100$', rotation=15,   fontsize=fs_sm, ha='center', va='center')

# Image plot
im = plt.imshow(r2gGR_full, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap=cmap, \
                vmin=r2gGR_min, vmax=r2gGR_max, interpolation='nearest', aspect='auto')
#r_edges = np.linspace(r_min, r_max, Nr_full)
#i_edges = np.linspace(i_max, i_max, Ni_full)
#x_mesh, y_mesh = np.meshgrid(r_edges, i_edges)
#im = ax.pcolormesh(x_mesh, y_mesh, gGR_full, cmap=cmap, norm=yx_norm)  # <-- NOT WORKING?!

# Colorbar
cmaj  = 50
ticks = [0,25,50,75,100,125]  #yx_cents[::cmaj]
CBgap = 0.033
Lcb = Lgap + xbox + CBgap
Bcb = Bgap
Wcb = 0.05
Hcb = 1.0 - (Bgap + Tgap)
cbar_ax = fig.add_axes([Lcb, Bcb, Wcb, Hcb])
cb = plt.colorbar(im, cax=cbar_ax, cmap=cmap, norm=yx_norm, ticks=ticks, boundaries=yx_edges, orientation='vertical', spacing='proportional', format='%2.2g')
cbar_ax.tick_params('y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
cbar_ax.tick_params('y', direction='out', length=tlmin, width=lw, which='minor')
cbar_ax.set_ylabel(r"$r_{\rm in}^{2} g_{\rm GR}$", fontsize=fs, ha='center', va='center', rotation=0)
#cbar_ax.yaxis.set_label_position('right')
cbar_ax.yaxis.set_label_coords(3.25, 0.5, transform=cbar_ax.transAxes)#fig.transFigure
cbar_ax.yaxis.tick_right()
cbar_ax.minorticks_off()

# Plot the black hole spin values on the top axis
Nticks_a = 5
a_tick_values = np.linspace(1, -1, Nticks_a)  # a values from -1 --> 1
a_tick_locations = np.zeros(Nticks_a)
for i in np.arange(Nticks_a):
    a_tick_locations[i] = (spin2risco(a=a_tick_values[i]) - x_min) / (x_max - x_min)  # rin values
a_tick_labels = [r"$1$", r"$0.5$", r"$0$", r"$-0.5$", r"$-1$"]
ax_top = ax.twiny()
ax_top.tick_params(axis='x', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tick_locations)
ax_top.set_xticklabels(a_tick_labels)
a_tickmin_locations = [(spin2risco(a=0.9) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.9) - x_min) / (x_max - x_min)]
ax_top.tick_params(axis='x', direction='in', length=tlmin, width=lw, which='minor', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tickmin_locations, minor=True)
ax_top.set_xticklabels(['0.9','','','','','','','','','','','','','','',''], minor=True)
ax_top.set_xlabel(r"$\leftarrow a$", fontsize=fs, ha='left', va='top', labelpad=0, x=1.0+CBgap+Wcb)

# Do this here to avoid the code snippet above messing with the y-axis for some reason
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()

#====================================================================================================
# PLOT gNT(r) and d/dr[gNT(r)]

fout = dout + 'gNT.eps'
print("")
print("Min/Max      gNT  = ", np.min(gNT),     np.max(gNT))
print("Min/Max d/dr[gNT] = ", np.min(dgNT_dr), np.max(dgNT_dr))

r_min,   r_max   = 1.0, 9.0
gNT_min, gNT_max = 0.5, 1.0
dgNT_dr_min, dgNT_dr_max = -0.5, 0.0

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 5.6
Lgap, Rgap, Bgap, Tgap = 0.2, 0.2, 0.225, 0.125
left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
xbox = 1.0 - (Lgap + Rgap)
ybox = 1.0 - (Bgap + Tgap)
fs, fs_sm, fs_leg, lw, pad, tlmaj, tlmin = 28, 24, 20, 2, 10, 10, 5
xlab_pos = (0.5, -0.2)
ylab_pos = (-0.225, 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.5

# Axes limits and tick increments
x_min, x_max = r_min, r_max
y_min, y_max = gNT_min, gNT_max
xinc_maj, xinc_min = 1, 0.5
yinc_maj, yinc_min = 0.1, 0.05
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = plt.axes([Lgap, Bgap, xbox, ybox])
ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
ax.set_ylabel(r"$g_{\rm NT}$", fontsize=fs, color='C0', ha='center', va='center', rotation=0)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params(axis='y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad, color='C0', labelcolor='C0')
ax.tick_params(axis='y', direction='out', length=tlmin, width=lw, which='minor', color='C0')
ax.tick_params(axis='x', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params(axis='x', direction='out', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
#ax.xaxis.set_label_position('top')

# Plot gNT(r)
ax.plot(r_grid, gNT, linewidth=lw, color='C0')

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(0.5, xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], 0.5, transform=ax.transAxes)#fig.transFigure


# Plot the black hole spin values on the top axis
Nticks_a = 5
a_tick_values = np.linspace(1, -1, Nticks_a)  # a values from -1 --> 1
a_tick_locations = np.zeros(Nticks_a)
for i in np.arange(Nticks_a):
    a_tick_locations[i] = (spin2risco(a=a_tick_values[i]) - x_min) / (x_max - x_min)  # rin values
a_tick_labels = [r"$1$", r"$0.5$", r"$0$", r"$-0.5$", r"$-1$"]
ax_top = ax.twiny()
ax_top.tick_params(axis='x', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tick_locations)
ax_top.set_xticklabels(a_tick_labels)
a_tickmin_locations = [(spin2risco(a=0.9) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.1) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=-0.9) - x_min) / (x_max - x_min)]
ax_top.set_xticks(a_tickmin_locations, minor=True)
ax_top.set_xticklabels([r"$0.9$",'','','','','','','','','','','','','','',''], minor=True)
ax_top.tick_params(axis='x', direction='out', length=tlmin, width=lw, which='minor', labelsize=fs_sm, pad=pad+tlmin)
ax_top.set_xlabel(r"$\leftarrow a$", fontsize=fs, ha='right', va='top', labelpad=0, x=1.0+Rgap+0.02)

# Draw the x-grid
a_list = np.linspace(-1.0, 1.0, 21).tolist()
for a_draw in a_list:
    r_draw = spin2risco(a=a_draw)
    ax.plot([r_draw,r_draw], [y_min,y_max], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)


axR = ax.twinx()
axR.set_xlim(x_min, x_max)
axR.set_ylim(dgNT_dr_min, dgNT_dr_max)
axR.plot(r_grid, dgNT_dr, linestyle='dashed', linewidth=lw, color='C1')
axR.tick_params(axis='y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad, color='C1')
axR.tick_params(axis='y', direction='out', length=tlmin, width=lw, which='minor', color='C1')
axR.set_ylabel(r"$\frac{d g_{\rm NT}}{d r_{\rm in}}$", fontsize=fs, color='C1', ha='center', va='center', rotation=0)

axR.spines['left'].set_color('C0')
axR.spines['right'].set_color('C1')
axR.tick_params(axis='y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad, color='C1', labelcolor='C1')
axR.tick_params(axis='y', direction='out', length=tlmin, width=lw, which='minor', color='C1')
axR.yaxis.set_label_coords(1.27, 0.5, transform=axR.transAxes)#fig.transFigure
axR.set_yticks([-0.45, -0.35, -0.25, -0.15, -0.05], minor=True)

# Do this here to avoid the code snippet above messing with the y-axis for some reason
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()
