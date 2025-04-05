"""
Plotting Functions
Functions that generate the nsigma bar plots per level.
"""

# TODO: change parameter description to colored
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

from ..w_transform import HaarTransform, InvHaarTransform

from .plottingtools import _BinData,  _findmin, _findmax
from .plottingtools import  _NewColorMap, _NSigmaFilter

__all__ = ['nsigScalogram', 'nsigFixedRes']

data_color='#0782B0'
nsigma_color='#54B959'

ymin_delta = 0.2 # small number

def nsigScalogram(data, hypothesis, nsigma, generating_function=None,
                  nsigma_min=None, nsigma_percent=1, reconstruction_scaled=False,
                  nsigma_colorcode=False, title=None, xlabel=None, outputfile=None):
    """
    Function that plots the nsigma values of the wavelet coefficients per level.
    Parameters
    ----------
    data : array
    Array to treat as the signal in the signal reconstruction.
    hypothesis : array
    Array to treat as the background in the signal reconstruction.
    nsigma : array
    Nsigma array of the data.
    nsigma_min : float
    Minimum value of the nsigmas to use for the signal reconstruction.
    nsigma_percent : float
    Percent of the decreasing-order nsigma array to use in the signal reconstruction.
    nsigma_colorcode : array
    Array of shape (nsigma) to use as the color-code for the bins. If None, the
    right and left side of the bins will be colored slightly different.
    """

    nsigCcoeffs = nsigma
    Level = len(nsigCcoeffs)-1

    data_hist, _, data_center, data_width = _BinData(data, bins=2**(Level))
    back_hist, _, back_center, back_width = _BinData(hypothesis, bins=2**(Level))

    cut = '(No cut)'
    if nsigma_percent is not None:
        cut = str(nsigma_percent*100) + '%'
    if nsigma_min is not None:
        cut = r'$\sigma_{min}$ = ' + str(nsigma_min)

    DeltaCoeff = _NSigmaFilter(data, hypothesis, nsigma, nsigma_min, nsigma_percent)

    ReconstructedData = InvHaarTransform(DeltaCoeff, normalize=False)
    if reconstruction_scaled is True:
        RecData = np.divide(ReconstructedData, np.sqrt(hypothesis))
    else:
        RecData = ReconstructedData

    nrows = Level+2
    ratio = [1.5] + [1.5]
    ratio += [1]*(Level)

    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(ncols=1, nrows=nrows,
                           height_ratios=ratio,
                           hspace=0)
    axs = [fig.add_subplot(gs[i,0]) for i in range(nrows)]

    # Fill out top panel
    axs[0].bar(data_center, data_hist, align='center',
               width=data_width, color=data_color, label="Data")

    axs[0].legend(edgecolor="black", fancybox=False, fontsize=12,
                  handlelength=0, handletextpad=0)
    axs[0].set_yscale('log')

    # The second panel will have the reconstructed signal
    if generating_function is not None:
        #norm = np.linalg.norm(hypothesis)
        generating_function = np.divide(generating_function, np.sqrt(hypothesis))
        signal_hist, _, signal_center, signal_width = _BinData(generating_function, bins=2**(Level))
        #axs[1].plot(signal_center, signal_hist, '*', markersize=3, color='red')
        axs[1].plot(signal_center, signal_hist, color='red', label="Generating Function")

    # RecData = np.divide(RecData, np.sqrt(hypothesis))
    axs[1].plot(data_center, RecData, 'o', markersize=3, color='#E67E22', label='Reconstruction ({})'.format(cut))

    axs[1].plot(range(len(data_center)), np.zeros_like(RecData), color='black', linewidth=0.5)
    axs[1].set_yscale('linear')
    axs[1].legend(edgecolor="black", fancybox=False, fontsize=12, handletextpad=0, handlelength=0, markerscale=0)


    coeffs_min = _findmin(nsigma[:Level]) # Use to set the min ylim of the plots
    coeffs_max = _findmax(nsigma[:Level]) # Use to set the max ylim of the plots

    cmap = _NewColorMap()
    sig_max = _findmax(np.absolute(nsigma[:Level]))
    norm = Normalize(vmin=0, vmax=sig_max)

    for l in range(Level):
        bins = 2**(Level-l-1)
        hist, edges, center, width = _BinData(nsigma[l], bins=bins)
        midLeft = (center-edges[:-1])/2.0
        LeftCenter = edges[:-1]+midLeft
        midRight = (edges[1:]-center)/2.0
        RightCenter = center+midRight

        if nsigma_colorcode==True:
            norm_points = norm(np.absolute(nsigma[l]))
            color_points = [cmap(i) for i in norm_points]
            axs[l+2].bar(LeftCenter, hist, align='center', width=width/2.0,
                         color=color_points)
            axs[l+2].bar(RightCenter, hist, align='center', width=width/2.0,
                         color=color_points, alpha=0.8)

        else:
            axs[l+2].bar(LeftCenter, hist, align='center', width=width/2.0,
                       color='#58C85E')
            axs[l+2].bar(RightCenter, hist, align='center', width=width/2.0,
                       color='#43A961')

        axs[l+2].set_ylim(coeffs_min-ymin_delta, coeffs_max+ymin_delta)
        #axs[l+2].plot(range(bins), np.zeros(bins), color='black', linewidth=0.5)
        axs[l+2].plot(edges, np.zeros_like(edges), color='black', linewidth=0.5)
        axs[l+2].tick_params(axis='x', bottom=False, labelbottom=False)
        lev = Level-l-1
        axs[l+2].text(x=-.062, y=.66, s=r'$N\sigma(\ell={%.1i})$'%(lev+1), fontsize=14,
                      #bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                      transform=axs[l+2].transAxes,
                      rotation=90)

    if nsigma_colorcode==True:
        cbar_axs = fig.add_axes([0.93, 0.15, 0.02, 0.7]) # colorbar axis
        cbar = ColorbarBase(cbar_axs, cmap=cmap, norm=norm)
        fig.text(x=0.93, y=.86, s=r'$N\sigma$', fontsize=12)

    if title is not None:
        fig.suptitle(title, fontsize=18, y=0.92)
        fig.text(x=0.5, y=0.1, s=xlabel, fontsize=14)
    if outputfile is not None:
        plt.savefig(outputfile, bbox_inches='tight')
    plt.show()




def nsigFixedRes(data, hypothesis, nsigma, nsigma_fixedres,
                 nsigma_colorcode=False, title=None, xlabel=None, outputfile=None):
    """
    Function that plots the nsigma values of the wavelet coefficients per level.
    The global significance per level is also displayed.
    Parameters
    ----------
    data : array
    Array to treat as the signal in the signal reconstruction.
    hypothesis : array
    Array to treat as the background in the signal reconstruction.
    nsigma : array
    Nsigma array of the data.
    nsigma_min : float
    Minimum value of the nsigmas to use for the signal reconstruction.
    nsigma_percent : float
    Percent of the decreasing-order nsigma array to use in the signal reconstruction.
    colorcode : array
    Array of shape (nsigma) to use as the color-code for the bins. If None, the
    right and left side of the bins will be colored slightly different.
    """

    nsigCcoeffs = nsigma
    Level = len(nsigCcoeffs)-1

    data_hist, _, data_center, data_width = _BinData(data, bins=2**(Level))
    back_hist, _, back_center, back_width = _BinData(hypothesis, bins=2**(Level))

    nrows = Level+1
    ratio = [1.5]
    ratio += [1]*(Level)

    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(ncols=2, nrows=nrows,
                           height_ratios=ratio,
                           width_ratios=[6,1],
                           hspace=0, wspace=0)
    axs = [fig.add_subplot(gs[i,0]) for i in range(nrows)]
    axs2 = [fig.add_subplot(gs[i,1]) for i in range(nrows)]

    # Fill out top panel
    axs[0].bar(data_center, data_hist, align='center',
               width=data_width, color=data_color, label="Data")
    #axs[0].text(x=.94, y=.63, s='Data', fontsize=12,
    #            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
    #            transform=axs[0].transAxes)
    axs[0].set_yscale('log')
    axs[0].legend(edgecolor="black", fancybox=False, fontsize=12,
                  handlelength=0, handletextpad=0)
    axs2[0].tick_params(axis='both', bottom=False, left=False,
                        labelbottom=False, labelleft=False)

    coeffs_min = _findmin(nsigma[:Level]) # Use to set the min ylim of the plots
    coeffs_max = _findmax(nsigma[:Level]) # Use to set the max ylim of the plots

    cmap = _NewColorMap()
    sig_max = _findmax(np.absolute(nsigma[:Level]))
    norm = Normalize(vmin=0, vmax=sig_max)

    sig_min = _findmin(nsigma_fixedres[:Level])
    sig_max = _findmax(nsigma_fixedres[:Level])

    for l in range(Level):
        bins = 2**(Level-l-1)
        hist, edges, center, width = _BinData(nsigma[l], bins=bins)
        midLeft = (center-edges[:-1])/2.0
        LeftCenter = edges[:-1]+midLeft
        midRight = (edges[1:]-center)/2.0
        RightCenter = center+midRight

        if nsigma_colorcode==True:
            norm_points = norm(np.absolute(nsigma[l]))
            color_points = [cmap(i) for i in norm_points]
            axs[l+1].bar(LeftCenter, hist, align='center', width=width/2.0,
                         color=color_points)
            axs[l+1].bar(RightCenter, hist, align='center', width=width/2.0,
                         color=color_points, alpha=0.8)

        else:
            axs[l+1].bar(LeftCenter, hist, align='center', width=width/2.0, color='#58C85E')
            axs[l+1].bar(RightCenter, hist, align='center', width=width/2.0, color='#43A961')

        axs[l+1].set_ylim(coeffs_min-ymin_delta, coeffs_max+ymin_delta)
        #axs[l+1].plot(range(bins), np.zeros(bins), color='black', linewidth=0.5)
        axs[l+1].plot(edges, np.zeros_like(edges), color='black', linewidth=0.5)
        axs[l+1].tick_params(axis='x', bottom=False, labelbottom=False)
        lev = Level-l-1
        axs[l+1].text(x=-.075, y=.66, s=r'$N\sigma(\ell={%.1i})$'%(lev+1), fontsize=14,
                      #bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},
                      transform=axs[l+1].transAxes, rotation=90)

        # -- Fill the right panels with the global significance per level --
        axs2[l+1].bar(0, nsigma_fixedres[l], color='#0A700F')
        axs2[l+1].tick_params(axis='x', bottom=False, labelbottom=False, labelleft=False)
        axs2[l+1].ticklabel_format(axis='y', style='sci')
        axs2[l+1].yaxis.tick_right()
        axs2[l+1].set_ylim(bottom=0.0, top=sig_max+ymin_delta)

    if nsigma_colorcode==True:
        cbar_axs = fig.add_axes([0.94, 0.15, 0.02, 0.7]) # colorbar axis
        cbar = ColorbarBase(cbar_axs, cmap=cmap, norm=norm)
        #cbar_axs.text(.5, sig_max*1.2, r'$N\sigma$', fontsize=12)
        fig.text(x=0.94, y=.86, s=r'$N\sigma$', fontsize=12)

    if title is not None:
        fig.suptitle(title, fontsize=18, y=0.92)
        fig.text(x=0.5, y=0.1, s=xlabel, fontsize=14)
    if outputfile is not None:
        plt.savefig(outputfile, bbox_inches='tight')
        plt.show()
