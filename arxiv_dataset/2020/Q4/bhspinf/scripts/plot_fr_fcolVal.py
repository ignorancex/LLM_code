import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from pdf_trans import *
from gs_stats import *

'''
python plot_fr_fcolVal.py
'''

#====================================================================================================
# Output file names
outdir     = '../ms/figures/'
#fout_LMCX1 = outdir + 'fr_fcol_LMCX1.eps'
#fout_U1543 = outdir + 'fr_fcol_U1543.eps'
fout_J1655 = outdir + 'fr_fcol_J1655.eps'
#fout_J1550 = outdir + 'fr_fcol_J1550.eps'
#fout_M33X7 = outdir + 'fr_fcol_M33X7.eps'
#fout_LMCX3 = outdir + 'fr_fcol_LMCX3.eps'
#fout_H1743 = outdir + 'fr_fcol_H1743.eps'
#fout_A0620 = outdir + 'fr_fcol_A0620.eps'

# HDF5 files containing f_r(r)
resdir   = '../data/fr/'
#fr_LMCX1 = resdir + 'fr_LMCX1.h5'
#fr_U1543 = resdir + 'fr_U1543.h5'
fr_J1655 = resdir + 'fr_J1655.h5'
#fr_J1550 = resdir + 'fr_J1550.h5'
#fr_M33X7 = resdir + 'fr_M33X7.h5'
#fr_LMCX3 = resdir + 'fr_LMCX3.h5'
#fr_H1743 = resdir + 'fr_H1743.h5'
#fr_A0620 = resdir + 'fr_A0620.h5'

#====================================================================================================
def bhspin_val_err(r, f_r):
    r_ml, y_ml, r_valm, y_valm, r_valp, y_valp = confidence_interval(x=r, y=f_r, sigma=0.683)
    a_val  = risco2spin(r_ml)
    a_errm = a_val - risco2spin(r_valp)  # Careful, a_min (a_max) corresponds to r_max (r_min)
    a_errp = risco2spin(r_valm) - a_val
    return a_val, a_errm, a_errp

#====================================================================================================
# PLOT: f_r(r) w/ different fcol value choices
def plot_fr_fcolVal(fout, fh5, source, aCF, xlims=None, ylims=None, xinc_maj=0.2, xinc_min=0.05, yinc_maj=2, yinc_min=0.5, Nminor_y=5):
    
    # Collect the f_r(r) results
    f = h5py.File(fh5, 'r')
    fcolCF   = f.get('fcolCF')[()]
    rCF      = f.get('rCF')[:]
    f_rCF    = f.get('f_rCF')[:]
    r        = f.get('r')[:]
    f_r      = f.get('f_r')[:]
    fcol_val = f.get('fcol_val')[:]
    rVal     = f.get('rVal')[:]
    f_rVal   = f.get('f_rVal')[:]
    r_min    = f.get('r_min')[()]
    r_max    = f.get('r_max')[()]
    f.close()
    print(source)

    # Normalization enforcement
    f_rCF /= np.trapz(y=f_rCF, x=rCF)
    f_r   /= np.trapz(y=f_r, x=r)
    for i in range(len(f_rVal)):
        f_rVal[i] /= np.trapz(y=f_rVal[i], x=rVal[i])

    Nfcol = np.size(fcol_val)

    # Plotting preferences
    dpi = 300
    xsize, ysize = 8.4, 5.6
    Lgap, Rgap, Bgap, Tgap = 0.15, 0.05, 0.175, 0.10
    Rgap = 1.0 - (1.0 - (Bgap + Tgap)) * (ysize/xsize) * 1.4 - Lgap  # 1.25 Square?
    left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
    xbox = 1.0 - (Lgap + Rgap)
    ybox = 1.0 - (Bgap + Tgap)
    fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
    xlab_pos = (0.5, -0.125)
    ylab_pos = (1.33*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 0.5)
    thin  = lw
    thick = lw * 2.0
    alpha = 0.5

    # Axes limits and tick increments
    x_min, x_max = xlims
    y_min, y_max = ylims
    if (y_min is None): y_min = -0.1 * y_max
    xmajorLocator = MultipleLocator(xinc_maj)
    xminorLocator = MultipleLocator(xinc_min)
    ymajorLocator = MultipleLocator(yinc_maj)
    yminorLocator = MultipleLocator(yinc_min)

    # Colormap, Bounds, and Normalization
    fcol_min, fcol_max, Ncb = 0.9, 2.5, 9
    cmap       = cm.get_cmap('viridis_r')
    fcol_edges = np.linspace(fcol_min, fcol_max, Ncb)
    fcol_cents = 0.5 * (fcol_edges[:-1] + fcol_edges[1:])
    fcol_norm  = BoundaryNorm(fcol_edges, cmap.N)
    s_map      = cm.ScalarMappable(norm=fcol_norm, cmap=cmap)

    # Setup the figure
    fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
    ax.set_ylabel(r"${\rm Marginal\ Density},\ f_{r}(r_{\rm in})$", fontsize=fs, ha='center', va='bottom')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params('both', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
    ax.tick_params('both', direction='in', length=tlmin, width=lw, which='minor')
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    # Align the axes labels (using the axes coordinate system)
    ax.xaxis.set_label_coords(xlab_pos[0], xlab_pos[1], transform=ax.transAxes)#fig.transFigure
    ax.yaxis.set_label_coords(ylab_pos[0], ylab_pos[1], transform=ax.transAxes)#fig.transFigure

    #....................................................................................................
    # Plot
    ax.plot(rCF, f_rCF, linewidth=thin, linestyle='dashed', color='k', zorder=1)

    color = []
    a_val, a_errm, a_errp = np.zeros(Nfcol), np.zeros(Nfcol), np.zeros(Nfcol)
    for i in range(Nfcol):
        
        # Plot
        color.append(cmap(fcol_norm(fcol_val[i])))
        ax.plot(rVal[i], f_rVal[i], linewidth=thin, linestyle='solid', color=color[i], zorder=1)

        # Collect the BH spin measurement
        a_val[i], a_errm[i], a_errp[i] = bhspin_val_err(r=rVal[i], f_r=f_rVal[i])

    aCF_val, aCF_errm, aCF_errp = aCF

    # f_r = 0 line
    ax.plot([x_min,x_max], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=2)

    #....................................................................................................
    # Print out the source name
    dy = 0.1
    dx = dy * (ybox / xbox) * (ysize / xsize)
    ax.text(1.0-dx, 1.0-dy, source, transform=ax.transAxes, color='k', fontsize=fs, ha='right', va='top')

    a_txt = []
    for i in range(Nfcol):
        a_txt.append(r"$" + '{:.2f}'.format(a_val[i]) + "^{+" + '{:.2f}'.format(a_errp[i]) + "}_{-" + '{:.2f}'.format(a_errm[i]) + "}$")

    # Different columns
    a1_txt, a2_txt, a3_txt = a_txt[0:2], a_txt[2:4], a_txt[4:]
    color1, color2, color3 = color[0:2], color[2:4], color[4:]
    # Append a2_txt and color2 with the CF result
    a2_txt.append(r"$" + '{:.2f}'.format(aCF_val) + "^{+" + '{:.2f}'.format(aCF_errp) + "}_{-" + '{:.2f}'.format(aCF_errm) + "}$")
    color2.append('k')
    # Print out the BH spin measurements
    dx1,   dy1   = dx*8.25,  dy*1.5
    xloc1, yloc1 = 1.0-dx1, 1.0-dy1
    for i in range(len(a1_txt)):
        ax.text(xloc1, yloc1-dy1, a1_txt[i], transform=ax.transAxes, color=color1[i], fontsize=fs_sm, ha='right', va='center')
        yloc1 -= dy1
    dx2,   dy2   = dx*4.75,  dy*1.5
    xloc2, yloc2 = 1.0-dx2, 1.0-dy2
    for i in range(len(a2_txt)):
        ax.text(xloc2, yloc2-dy2, a2_txt[i], transform=ax.transAxes, color=color2[i], fontsize=fs_sm, ha='right', va='center')
        yloc2 -= dy2
    dx3,   dy3   = dx*1.0,  dy*1.5
    xloc3, yloc3 = 1.0-dx3, 1.0-dy3
    for i in range(len(a3_txt)):
        ax.text(xloc3, yloc3-dy3, a3_txt[i], transform=ax.transAxes, color=color3[i], fontsize=fs_sm, ha='right', va='center')
        yloc3 -= dy3

    #....................................................................................................
    # Plot the black hole spin values on the top axis
    Nticks_a         = 5
    a_tick_values    = np.linspace(1, -1, Nticks_a)  # a values from 0 --> 1
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
    ax_top.set_xlabel(r"$\leftarrow a$", fontsize=fs, ha='right', va='top', labelpad=0, x=1)

    # Draw the x-grid
    a_list = np.linspace(-1, 1, 21).tolist()
    for a_draw in a_list:
        r_draw = spin2risco(a=a_draw)
        ax.plot([r_draw,r_draw], [y_min,y_max], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)

    # Colorbar
    cmaj  = 1#2
    CBgap = 0.025
    Lcb = Lgap + xbox + CBgap
    Bcb = Bgap
    Wcb = 0.05
    Hcb = 1.0 - (Bgap + Tgap)
    cbar_ax = fig.add_axes([Lcb, Bcb, Wcb, Hcb])
    cb = plt.colorbar(s_map, cax=cbar_ax, cmap=cmap, norm=fcol_norm, ticks=fcol_cents[::cmaj], boundaries=fcol_edges, orientation='vertical', spacing='proportional', format='%2.1f')
    cbar_ax.tick_params('y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
    cbar_ax.tick_params('y', direction='out', length=tlmin, width=lw, which='minor')
    cbar_ax.set_xlabel(r"$f_{\rm col}$", fontsize=fs, ha='center', va='center', rotation=0)
    #cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_label_coords(2.15, 1.05, transform=cbar_ax.transAxes)#fig.transFigure
    #cbar_ax.yaxis.set_label_position('right')
    #cbar_ax.yaxis.set_label_coords(3.5, 0.5, transform=cbar_ax.transAxes)#fig.transFigure
    cbar_ax.yaxis.tick_right()
    cbar_ax.minorticks_off()

    # Do this here to avoid the code snippet above messing with the y-axis for some reason
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    # Update the y-axis to scientific notation
    #ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # !!! Seems to fix the issue of the different fonts !!!
    plt.rcParams.update({'font.family': 'custom'})

    # Save the figure
    fig.savefig(fout, bbox_inches=0, dpi=dpi)
    plt.close()

#====================================================================================================
# PLOTS

xlims    = [0,12]
xinc_maj = 2
xinc_min = 1

# LMC X-1
#plot_fr_fcolVal(fout=fout_LMCX1, fh5=fr_LMCX1, aCF=[0.92, 0.07, 0.05], source=r"LMC X--1",
#        xlims=xlims, ylims=[None,2.25], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.5, yinc_min=0.25)

# 4U 1543-47
#plot_fr_fcolVal(fout=fout_U1543, fh5=fr_U1543, aCF=[0.80, 0.10, 0.10], source=r"4U 1543--47",
#        xlims=xlims, ylims=[None,1.75], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.5, yinc_min=0.25)

# GRO J1655-40
plot_fr_fcolVal(fout=fout_J1655, fh5=fr_J1655, aCF=[0.70, 0.10, 0.10], source=r"GRO J1655--40",
        xlims=xlims, ylims=[None,3.25], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.5, yinc_min=0.25)

# XTE J1550-564
#plot_fr_fcolVal(fout=fout_J1550, fh5=fr_J1550, aCF=[0.34, 0.28, 0.20], source=r"XTE J1550--564",
#        xlims=xlims, ylims=[None,1], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1)

# M33 X-7
#plot_fr_fcolVal(fout=fout_M33X7, fh5=fr_M33X7, aCF=[0.84, 0.05, 0.05], source=r"M33 X--7",
#        xlims=xlims, ylims=[None,4.5], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=1, yinc_min=0.5)

# LMC X-3
#plot_fr_fcolVal(fout=fout_LMCX3, fh5=fr_LMCX3, aCF=[0.25, 0.16, 0.13], source=r"LMC X--3",
#        xlims=xlims, ylims=[None,2.25], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.5, yinc_min=0.25)

# H1743-322
#plot_fr_fcolVal(fout=fout_H1743, fh5=fr_H1743, aCF=[0.20, 0.33, 0.34], source=r"H1743--322",
#        xlims=xlims, ylims=[None,1], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1)

# A0620-00
#plot_fr_fcolVal(fout=fout_A0620, fh5=fr_A0620, aCF=[0.12, 0.19, 0.19], source=r"A0620--00",
#        xlims=xlims, ylims=[None,1.75], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.5, yinc_min=0.25)

#====================================================================================================
