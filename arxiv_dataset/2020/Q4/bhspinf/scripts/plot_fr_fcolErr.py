import numpy as np
import pylab as plt
import h5py
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter, ScalarFormatter
from gs_stats import confidence_interval
from pdf_trans import spin2risco, risco2spin
from scipy.interpolate import interp1d
from PIL import Image

'''
python plot_fr_fcolErr.py
'''

#====================================================================================================
# Output file names
outdir     = '../ms/figures/'
fout_LMCX1 = outdir + 'fr_LMCX1.eps'
fout_U1543 = outdir + 'fr_U1543.eps'
fout_J1655 = outdir + 'fr_J1655.eps'
fout_J1550 = outdir + 'fr_J1550.eps'
fout_M33X7 = outdir + 'fr_M33X7.eps'
fout_LMCX3 = outdir + 'fr_LMCX3.eps'
fout_H1743 = outdir + 'fr_H1743.eps'
fout_A0620 = outdir + 'fr_A0620.eps'

# HDF5 files containing f_r(r)
resdir   = '../data/fr/'
fr_LMCX1 = resdir + 'fr_LMCX1.h5'
fr_U1543 = resdir + 'fr_U1543.h5'
fr_J1655 = resdir + 'fr_J1655.h5'
fr_J1550 = resdir + 'fr_J1550.h5'
fr_M33X7 = resdir + 'fr_M33X7.h5'
fr_LMCX3 = resdir + 'fr_LMCX3.h5'
fr_H1743 = resdir + 'fr_H1743.h5'
fr_A0620 = resdir + 'fr_A0620.h5'

#====================================================================================================
def bhspin_val_err(r, f_r):
    r_ml, y_ml, r_valm, y_valm, r_valp, y_valp = confidence_interval(x=r, y=f_r, sigma=0.683)
    a_val  = risco2spin(r_ml)
    a_errm = a_val - risco2spin(r_valp)  # Careful, a_min (a_max) corresponds to r_max (r_min)
    a_errp = risco2spin(r_valm) - a_val
    return a_val, a_errm, a_errp
    '''
    # SHOULD WE BE USING THE SPIN PDF INSTEAD? DOES IT MATTER?
    Nr = np.size(r)
    a = np.zeros(Nr)
    for i in range(Nr):
        a[i] = risco2spin(r[i])
    f_r_interp = interp1d(x=r, y=f_r, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
    f_r = generate_f_x(f_x_interp=f_r_interp)
    f_a = transform_frisco2fspin(f_r)
    f_a = f_a.pdf(a)
    a_ml, y_ml, a_valm, y_valm, a_valp, y_valp = confidence_interval(x=a, y=f_a, sigma=0.683)
    a_errm  = a_ml  - a_valm
    a_errp  = a_valp - a_ml
    a_val   = a_ml
    return a_val, a_errm, a_errp
    '''

#https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple/42658124#42658124
class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=False):
        self.oom     = order
        self.fformat = fformat
        ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

#====================================================================================================
# PLOT: f_r(r) w/ different fcol uncertainty choices
def plot_fr(fout, fh5, source, aCF, xlims=None, ylims=None, xinc_maj=0.2, xinc_min=0.05, yinc_maj=2, yinc_min=0.5, Nminor_y=5, ylog=False, cutTop=False, cutBot=False):

    # Collect the f_r(r) results
    f = h5py.File(fh5, 'r')
    fcolCF   = f.get('fcolCF')[()]
    rCF      = f.get('rCF')[:]
    f_rCF    = f.get('f_rCF')[:]
    r        = f.get('r')[:]
    f_r      = f.get('f_r')[:]
    fcol_err = f.get('fcol_err')[:]
    rErr     = f.get('rErr')[:]
    f_rErr   = f.get('f_rErr')[:]
    r_min    = f.get('r_min')[()]
    r_max    = f.get('r_max')[()]
    f.close()
    print(source)

    # Normalization enforcement
    f_rCF /= np.trapz(y=f_rCF, x=rCF)
    f_r   /= np.trapz(y=f_r, x=r)
    for i in range(len(f_rErr)):
        f_rErr[i] /= np.trapz(y=f_rErr[i], x=rErr[i])
    
    # Print out the mean and standard deviation for the black hole spin
    def nmoment(x, counts, c, n):
        return np.sum(counts*(x-c)**n) / np.sum(counts)
    rCF_mean = nmoment(rCF,     f_rCF,     0, 1)
    r1_mean  = nmoment(rErr[0], f_rErr[0], 0, 1)
    r2_mean  = nmoment(rErr[1], f_rErr[1], 0, 1)
    r3_mean  = nmoment(rErr[2], f_rErr[2], 0, 1)
    aCF_mean = risco2spin(rCF_mean)
    a1_mean  = risco2spin(r1_mean)
    a2_mean  = risco2spin(r2_mean)
    a3_mean  = risco2spin(r3_mean)
    print("\t     aCF: Mean a = ", aCF_mean)
    print("\t +/- 0.1: Mean a = ", a1_mean)
    print("\t +/- 0.2: Mean a = ", a2_mean)
    print("\t +/- 0.3: Mean a = ", a3_mean)
    
    Nfcol = np.size(fcol_err)
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    # Plotting preferences
    dpi = 300
    xsize, ysize = 8.4, 5.6#4.2
    Lgap, Rgap, Bgap, Tgap = 0.125, 0.025, 0.225, 0.125
    left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
    xbox = 1.0 - (Lgap + Rgap)
    ybox = 1.0 - (Bgap + Tgap)
    fs, fs_sm, fs_leg, lw, pad, tlmaj, tlmin = 28, 24, 20, 2, 10, 10, 5
    xlab_pos = (0.5, -0.2)
    ylab_pos = (-0.075, 0.5*ybox+Bgap)
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

    # Setup
    fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
    ax.set_ylabel(r"$f_{r}( r_{\rm in} )$", fontsize=fs, ha='center', va='bottom')
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
    # Plot f_rCF(r) and the "recovered" f_r(r)
    ax.plot(rCF, f_rCF, linewidth=thick, linestyle='solid', color='black', zorder=1)
    ax.plot(r,   f_r,   linewidth=thin,  linestyle='solid', color='grey',  zorder=2)

    # BH spin measurement <-- w/o fcol uncertainties
    #aCF_val, aCF_errm, aCF_errp = bhspin_val_err(r=rCF, f_r=f_rCF)
    aCF_val, aCF_errm, aCF_errp = aCF

    #....................................................................................................
    # Plot f_r(r) w/ fcol uncertainties
    for i in range(Nfcol):
        ax.plot(rErr[i], f_rErr[i], linewidth=thin, linestyle='solid', color=color[i], zorder=3)
    
    # BH spin measurement <-- w/ fcol uncertainties
    a_val, a_errm, a_errp = np.zeros(Nfcol), np.zeros(Nfcol), np.zeros(Nfcol)
    for i in range(Nfcol):
        a_val[i], a_errm[i], a_errp[i] = bhspin_val_err(r=rErr[i], f_r=f_rErr[i])
    
    # f_r = 0 line
    ax.plot([x_min,x_max], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=4)

    #....................................................................................................
    # Print out the source name
    dy = 0.1
    dx = dy * (ybox / xbox) * (ysize / xsize)
    ax.text(1.0-dx, 1.0-dy, source, transform=ax.transAxes, color='k', fontsize=fs, ha='right', va='top')

    dy  *= 1.5#1.75
    yloc = 1.0-dy
    
    # Print out the aCF measurement...
    aCF_txt = r"$a = " + '{:.2f}'.format(aCF_val) + "^{+" + '{:.2f}'.format(aCF_errp) + "}_{-" + '{:.2f}'.format(aCF_errm) + "}$"
    ax.text(1.0-dx, yloc-dy, aCF_txt, transform=ax.transAxes, color='k', fontsize=fs_sm, ha='right', va='center')
    yloc -= dy
    
    # ...and the revised BH spin measurement
    for i in range(Nfcol):
        a_txt = r"$" + '{:.2f}'.format(a_val[i]) + "^{+" + '{:.2f}'.format(a_errp[i]) + "}_{-" + '{:.2f}'.format(a_errm[i]) + "}$"
        ax.text(1.0-dx, yloc-dy, a_txt, transform=ax.transAxes, color=color[i], fontsize=fs_sm, ha='right', va='center')
        yloc -= dy
    
    # Print % increase in uncertainty
    aerr_CF   = (aCF_errp  + aCF_errm)
    aerr_pct1 = (a_errp[0] + a_errm[0]) / aerr_CF
    aerr_pct2 = (a_errp[1] + a_errm[1]) / aerr_CF
    aerr_pct3 = (a_errp[2] + a_errm[2]) / aerr_CF
    print("\t +/- 0.1: a % Err = ", aerr_pct1)
    print("\t +/- 0.2: a % Err = ", aerr_pct2)
    print("\t +/- 0.3: a % Err = ", aerr_pct3)

    #....................................................................................................
    # Plot the black hole spin values on the top axis
    Nticks_a = 5
    a_tick_values = np.linspace(1, -1, Nticks_a)  # a values from -1 --> 1
    a_tick_locations = np.zeros(Nticks_a)
    for i in np.arange(Nticks_a):
        a_tick_locations[i] = (spin2risco(a=a_tick_values[i]) - x_min) / (x_max - x_min)  # rin values
    a_tick_labels = ['', r"$0.5$", r"$0$", r"$-0.5$", '']
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
    a_list = np.linspace(-1.0, 1.0, 21).tolist()
    for a_draw in a_list:
        r_draw = spin2risco(a=a_draw)
        ax.plot([r_draw,r_draw], [y_min,y_max], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)
    
    # Do this here to avoid the code snippet above messing with the y-axis for some reason
    if (ylog is True):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)

    # Update the y-axis to scientific notation
    ax.yaxis.set_major_formatter(OOMFormatter(-1, "%1.0f"))
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # If desired, turn off the top/bottom x-axes
    if (cutTop is True):
        ax_top.set_xlabel(None)
        ax_top.set_xticklabels([], minor=False)
        ax_top.set_xticklabels([], minor=True)
    if (cutBot is True):
        ax.set_xlabel(None)
        ax.set_xticklabels([], minor=False)
        ax.set_xticklabels([], minor=True)

    # !!! Seems to fix the issue of the different fonts !!!
    plt.rcParams.update({'font.family': 'custom'})

    # Crop the whitespace if necessary
    Tdy, Bdy = 0.05, 0.2
    if ( (cutTop is True ) and (cutBot is True ) ): t, b = top+Tdy, bottom-Bdy
    if ( (cutTop is True ) and (cutBot is False) ): t, b = top+Tdy, bottom
    if ( (cutTop is False) and (cutBot is True ) ): t, b = top,     bottom-Bdy
    plt.subplots_adjust(left=left, right=right, top=t, bottom=b)
    ysize_new = ysize * (top-bottom) / (t-b)
    fig.set_size_inches(xsize, ysize_new*1.001)
    
    # Save the figure
    fig.savefig(fout, bbox_inches=0, dpi=dpi)
    plt.close()

    '''
    # Crop the whitespace if necessary
    Tdy, Bdy = 0.067, 0.033
    im   = Image.open(fout)
    # Rasterize onto 4x higher resolution grid
    #im.load(scale=4.0)
    w, h = im.size
    l, r = 0, w
    t, b = 0, h
    if ( (cutTop is True ) and (cutBot is True ) ): t, b = (Tgap-Tdy)*h, h-(Bgap-Bdy)*h
    if ( (cutTop is True ) and (cutBot is False) ): t, b = (Tgap-Tdy)*h, h
    if ( (cutTop is False) and (cutBot is True ) ): t, b = 0, h-(Bgap-Bdy)*h
    im.crop((l, t, r, b)).save(fout)
    im.close()
    '''

#====================================================================================================
# PLOTS

xlims    = [0,12]
xinc_maj = 1
xinc_min = 0.5

# LMC X-1
plot_fr(fout=fout_LMCX1, fh5=fr_LMCX1, aCF=[0.92, 0.07, 0.05], source=r"LMC X--1",
        xlims=xlims, ylims=[None,1.1], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1, cutTop=False, cutBot=True)

# 4U 1543-47
plot_fr(fout=fout_U1543, fh5=fr_U1543, aCF=[0.80, 0.10, 0.10], source=r"4U 1543--47",
        xlims=xlims, ylims=[None,0.9], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1, cutTop=True, cutBot=True)

# GRO J1655-40
plot_fr(fout=fout_J1655, fh5=fr_J1655, aCF=[0.70, 0.10, 0.10], source=r"GRO J1655--40",
        xlims=xlims, ylims=[None,1], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1, cutTop=True, cutBot=False)

# XTE J1550-564
plot_fr(fout=fout_J1550, fh5=fr_J1550, aCF=[0.34, 0.28, 0.20], source=r"XTE J1550--564",
        xlims=xlims, ylims=[None,0.55], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.1, yinc_min=0.05, cutTop=False, cutBot=True)

# M33 X-7
plot_fr(fout=fout_M33X7, fh5=fr_M33X7, aCF=[0.84, 0.05, 0.05], source=r"M33 X--7",
        xlims=xlims, ylims=[None,1.6], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.4, yinc_min=0.2, cutTop=True, cutBot=True)

# LMC X-3
plot_fr(fout=fout_LMCX3, fh5=fr_LMCX3, aCF=[0.25, 0.16, 0.13], source=r"LMC X--3",
        xlims=xlims, ylims=[None,0.9], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1, cutTop=True, cutBot=True)

# H1743-322
plot_fr(fout=fout_H1743, fh5=fr_H1743, aCF=[0.20, 0.33, 0.34], source=r"H1743--322",
        xlims=xlims, ylims=[None,0.35], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.1, yinc_min=0.05, cutTop=True, cutBot=True)

# A0620-00
plot_fr(fout=fout_A0620, fh5=fr_A0620, aCF=[0.12, 0.19, 0.19], source=r"A0620--00",
        xlims=xlims, ylims=[None,0.8], xinc_maj=xinc_maj, xinc_min=xinc_min, yinc_maj=0.2, yinc_min=0.1, cutTop=True, cutBot=False)

#====================================================================================================
