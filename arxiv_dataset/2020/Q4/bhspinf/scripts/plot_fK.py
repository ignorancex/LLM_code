import numpy as np
import pylab as plt
import h5py
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter, ScalarFormatter, FuncFormatter
from gs_stats import confidence_interval
from scipy.interpolate import interp1d
from PIL import Image

'''
python plot_fK.py
'''

# Output file names
outdir     = '../ms/figures/'
fout_LMCX1 = outdir + 'fK_LMCX1.eps'
fout_U1543 = outdir + 'fK_U1543.eps'
fout_J1655 = outdir + 'fK_J1655.eps'
fout_J1550 = outdir + 'fK_J1550.eps'
fout_M33X7 = outdir + 'fK_M33X7.eps'
fout_LMCX3 = outdir + 'fK_LMCX3.eps'
fout_H1743 = outdir + 'fK_H1743.eps'
fout_A0620 = outdir + 'fK_A0620.eps'

# HDF5 files containing f_K(K)
resdir   = '../data/fK/'
fr_LMCX1 = resdir + 'fK_LMCX1.h5'
fr_U1543 = resdir + 'fK_U1543.h5'
fr_J1655 = resdir + 'fK_J1655.h5'
fr_J1550 = resdir + 'fK_J1550.h5'
fr_M33X7 = resdir + 'fK_M33X7.h5'
fr_LMCX3 = resdir + 'fK_LMCX3.h5'
fr_H1743 = resdir + 'fK_H1743.h5'
fr_A0620 = resdir + 'fK_A0620.h5'

#====================================================================================================

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
def plot_fK(fout, fh5, source, xlims=None, ylims=None, xinc_maj=0.2, xinc_min=0.05, yinc_maj=2, yinc_min=0.5, Nminor_y=5, ylog=False, cutTop=False, cutBot=False, yoom=None, xlab_rm=False):

    # Collect the f_K(K) results
    f = h5py.File(fh5, 'r')
    K     = f.get('K')[:]
    f_K   = f.get('lcf_K')[:]
    K_min = f.get('K_min')[()]
    K_max = f.get('K_max')[()]
    f.close()
    print(source)

    # Normalization enforcement
    f_K /= np.trapz(y=f_K, x=K)

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
    ax.set_xlabel(r"${\rm Disc\ Flux\ Normalization},\ K_{\rm flux}\ [{\rm km^{2} / (kpc / 10)^{2}}]$", fontsize=fs, ha='center', va='top')
    ax.set_ylabel(r"$f_{K}( K_{\rm flux} )$", fontsize=fs, ha='center', va='bottom')
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
    # Plot f_K(K)
    ax.plot(K, f_K, linewidth=thick, linestyle='solid', color='black', zorder=2)

    # Median and inter-68% range
    K_ml, y_ml, K_valm, y_valm, K_valp, y_valp = confidence_interval(x=K, y=f_K, sigma=0.683)
    ax.plot([K_ml,K_ml],     [0,y_ml],   linewidth=thin, linestyle='solid', color='black', zorder=1)
    ax.plot([K_valm,K_valm], [0,y_valm], linewidth=thin, linestyle='solid', color='black', zorder=1)
    ax.plot([K_valp,K_valp], [0,y_valp], linewidth=thin, linestyle='solid', color='black', zorder=1)
    i_K_valm = np.argmin(np.abs(K - K_valm))
    i_K_valp = np.argmin(np.abs(K - K_valp))
    N_fill   = i_K_valp+1 - i_K_valm
    y1_fill  = np.zeros(N_fill)
    y2_fill  = f_K[i_K_valm:i_K_valp+1]
    x_fill   = K[i_K_valm:i_K_valp+1]
    ax.fill_between(x=x_fill, y1=y1_fill, y2=y2_fill, alpha=alpha, facecolor='LightGray')

    # f_K = 0 line
    ax.plot([x_min,x_max], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=2)

    #....................................................................................................

    # Print out the source name
    dy = 0.1
    dx = dy * (ybox / xbox) * (ysize / xsize)
    ax.text(1.0-dx, 1.0-dy, source, transform=ax.transAxes, color='k', fontsize=fs, ha='right', va='top')

    dy  *= 1.5#1.75
    yloc = 1.0-dy
    
    # Print out the Kflux measurement...
    K_val  = K_ml
    K_errm = (K_ml - K_valm)
    K_errp = (K_valp - K_ml)
    Koom   = np.floor(np.log10(K_ml))
    if ((Koom > 2) or (Koom < 0)):
        K_txt  = r"$K_{\rm flux} = " + '{:3.2f}'.format(K_val/10**Koom) + "^{+" + '{:3.2f}'.format(K_errp/10**Koom) + "}_{-" + '{:3.2f}'.format(K_errm/10**Koom) + "}$"
        ax.text(1.0-dx, yloc-dy, K_txt, transform=ax.transAxes, color='k', fontsize=fs_sm, ha='right', va='center')
        yloc    -= dy
        Koom_txt = r"$\times 10^{" + str(int(Koom)) + "}$"
        ax.text(1.0-dx, yloc-dy, Koom_txt, transform=ax.transAxes, color='k', fontsize=fs_sm, ha='right', va='center')
    else:
        if (Koom ==  0): K_txt = r"$K_{\rm flux} = " + '{:3.2f}'.format(K_val) + "^{+" + '{:3.2f}'.format(K_errp) + "}_{-" + '{:3.2f}'.format(K_errm) + "}$"
        if (Koom ==  1): K_txt = r"$K_{\rm flux} = " + '{:3.1f}'.format(K_val) + "^{+" + '{:3.1f}'.format(K_errp) + "}_{-" + '{:3.1f}'.format(K_errm) + "}$"
        if (Koom ==  2): K_txt = r"$K_{\rm flux} = " + '{:3.0f}'.format(K_val) + "^{+" + '{:3.0f}'.format(K_errp) + "}_{-" + '{:3.0f}'.format(K_errm) + "}$"
        ax.text(1.0-dx, yloc-dy, K_txt, transform=ax.transAxes, color='k', fontsize=fs_sm, ha='right', va='center')
        
    #....................................................................................................

    # Update the y-axis to scientific notation
    if (yoom is None): yoom = np.floor(np.log10(y_max))
    ax.yaxis.set_major_formatter(OOMFormatter(yoom, "%1.0f"))
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    #xoom = np.floor(np.log10(x_max))
    #ax.xaxis.set_major_formatter(OOMFormatter(xoom, "%1.1f"))
    #ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Use commas for x-axis tick labels (e.g., 15,000)
    if (x_min >= 1e4): ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    # If desired, turn off the top/bottom x-axes
    if (cutBot is True):
        ax.set_xlabel(None)
        #ax.set_xticklabels([], minor=False)
        #ax.set_xticklabels([], minor=True)

    # If desired, remove the rightmost x-label <-- DOES NOT WORK!
    if (xlab_rm is True):
        x_labels = ax.get_xticklabels()
        x_labels[-1] = ""
        ax.set_xticklabels(x_labels)

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
    w, h = im.size
    l, r = 0, w
    t, b = 0, h
    #if ( (cutTop is True ) and (cutBot is True ) ): t, b = (Tgap-Tdy)*h, h-(Bgap-Bdy)*h
    #if ( (cutTop is True ) and (cutBot is False) ): t, b = (Tgap-Tdy)*h, h
    #if ( (cutTop is False) and (cutBot is True ) ): t, b = 0, h-(Bgap-Bdy)*h
    if ( (cutTop is True ) and (cutBot is True ) ): t, b = (Tgap-Tdy)*h, h-(Bgap-3*Bdy)*h
    if ( (cutTop is True ) and (cutBot is False) ): t, b = (Tgap-Tdy)*h, h
    if ( (cutTop is False) and (cutBot is True ) ): t, b = 0, h-(Bgap-3*Bdy)*h
    im.crop((l, t, r, b)).save(fout)
    im.close()
    '''
    
#====================================================================================================
# PLOTS

# LMC X-1
plot_fK(fout=fout_LMCX1, fh5=fr_LMCX1, source=r"LMC X--1", yoom=-1,
        xlims=[0,25], ylims=[None,0.3], xinc_maj=5, xinc_min=1, yinc_maj=0.1, yinc_min=0.025, cutTop=False, cutBot=True)

# 4U 1543-47
plot_fK(fout=fout_U1543, fh5=fr_U1543, source=r"4U 1543--47", yoom=-3,
        xlims=[0,600], ylims=[None,0.01], xinc_maj=100, xinc_min=50, yinc_maj=0.002, yinc_min=0.001, cutTop=True, cutBot=True)

# GRO J1655-40
plot_fK(fout=fout_J1655, fh5=fr_J1655, source=r"GRO J1655--40", yoom=-3,
        xlims=[0,800], ylims=[None,0.01], xinc_maj=100, xinc_min=50, yinc_maj=0.002, yinc_min=0.001, cutTop=True, cutBot=False)

# XTE J1550-564
plot_fK(fout=fout_J1550, fh5=fr_J1550, source=r"XTE J1550--564", yoom=-2,
        xlims=[0,1250], ylims=[None,0.05], xinc_maj=200, xinc_min=100, yinc_maj=0.01, yinc_min=0.005, cutTop=False, cutBot=True)

# M33 X-7
plot_fK(fout=fout_M33X7, fh5=fr_M33X7, source=r"M33 X--7", yoom=2,
        xlims=[0.005,0.0325], ylims=[None,350], xinc_maj=0.005, xinc_min=0.001, yinc_maj=100, yinc_min=50, cutTop=True, cutBot=True)

# LMC X-3
plot_fK(fout=fout_LMCX3, fh5=fr_LMCX3, source=r"LMC X--3", yoom=0,
        xlims=[3,6], ylims=[None,3.5], xinc_maj=0.5, xinc_min=0.1, yinc_maj=1.0, yinc_min=0.5, cutTop=True, cutBot=True)

# H1743-322
plot_fK(fout=fout_H1743, fh5=fr_H1743, source=r"H1743--322", yoom=-2,
        xlims=[0,250], ylims=[None,0.055], xinc_maj=50, xinc_min=25, yinc_maj=0.01, yinc_min=0.005, cutTop=True, cutBot=True)

# A0620-00
plot_fK(fout=fout_A0620, fh5=fr_A0620, source=r"A0620--00", yoom=-4,
        xlims=[15000,19999], ylims=[None,0.0015], xinc_maj=1000, xinc_min=250, yinc_maj=0.0005, yinc_min=0.00025, cutTop=True, cutBot=False)#, xlab_rm=True)

#====================================================================================================
