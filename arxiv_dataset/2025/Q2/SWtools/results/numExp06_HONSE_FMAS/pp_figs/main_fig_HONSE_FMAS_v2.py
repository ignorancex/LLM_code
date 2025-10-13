import sys
import os
import scipy.optimize as so
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def save_fig(fig_name='test', fig_format='png'):
    dir_name = os.path.dirname(fig_name)
    os.makedirs(dir_name,exist_ok=True)
    if fig_format == 'png':
        plt.savefig(fig_name+'.png', format='png', dpi=600)
    elif fig_format == 'pdf':
        plt.savefig(fig_name+'.pdf', format='pdf', dpi=600)
    elif fig_format == 'svg':
        plt.savefig(fig_name+'.svg', format='svg', dpi=600)
    else:
        plt.show()


def set_style(fig_width=3.25, aspect_ratio = 0.6):

    fig_height = aspect_ratio*fig_width

    params = {
        'figure.figsize': (fig_width,fig_height),
        'legend.fontsize': 6,
        'legend.frameon': False,
        'axes.labelsize': 7,
        'axes.linewidth': 1.,
        'axes.linewidth': 0.8,
        'xtick.labelsize' :7,
        'ytick.labelsize': 7,
        'mathtext.fontset': 'stixsans',
        'mathtext.rm': 'serif',
        'mathtext.bf': 'serif:bold',
        'mathtext.it': 'serif:italic',
        'mathtext.sf': 'sans\\-serif',
        'font.size':  7,
        'font.family': 'serif',
        'font.serif': "Helvetica",
    }
    mpl.rcParams.update(params)


def main():

    o_name = './fig_HONSE_FMAS_v2'
    o_format = 'png'

    def subfig_label(ax, label):
        pos = ax.get_position()
        fig.text(
            pos.x0,
            pos.y1,
            label,
            color="white",
            backgroundcolor="k",
            bbox=dict(facecolor="k", edgecolor="none", boxstyle="square,pad=0.1"),
            verticalalignment="top",
            horizontalalignment="left",
        )

    set_style(3.5, 1.5)
    fig = plt.figure()
    plt.subplots_adjust(left = 0.095, bottom = 0.07, right = 0.99, top = 0.98)
    gs00 = GridSpec(nrows = 7, ncols = 1, wspace=1., hspace=0.7)

    gsA = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs00[0:3,0], wspace=0.3, hspace=0.075)
    ax1 = fig.add_subplot(gsA[0, 0])
    ax2 = fig.add_subplot(gsA[0, 1])
    sf1=[ax1, ax2]

    gsB = GridSpecFromSubplotSpec(8, 1, subplot_spec=gs00[3:,0], wspace=0.3, hspace=0.2)
    ax1 = fig.add_subplot(gsB[0:4, 0])
    ax2 = fig.add_subplot(gsB[4:6, 0])
    ax3 = fig.add_subplot(gsB[6:, 0])
    sf2=[ax1, ax2, ax3]

    # -- SUBFIGURE CONTENT ----------------------------------------------------
    subfig_ab(fig, sf1)
    subfig_label(sf1[0],r"(a)")
    subfig_label(sf1[1],r"(b)")

    subfig_cd(fig, sf2)
    subfig_label(sf2[0],r"(c)")
    subfig_label(sf2[1],r"(d)")
    subfig_label(sf2[2],r"(e)")

    # -- GENERATE FIGURE ------------------------------------------------------
    save_fig(fig_name=o_name, fig_format=o_format )



def subfig_ab(fig, axs):
    ax1, ax2 = axs

    def _fetch_data(f_name):
        dat = np.load(f_name)
        xi = dat['xi']
        U = dat['U0']
        iter_list = dat['it']
        acc_list = dat['acc']
        return xi, U, iter_list, acc_list

    # -------

    xi, U, iter_list, acc_list = _fetch_data('../res.npz')
    ax1.plot(xi, np.real(U), color='C0', lw=1., dashes=[], label=r'${\rm{Re}}[U]$')
    ax1.plot(xi, np.imag(U), color='C0', lw=1., dashes=[2,1], label=r'${\rm{Im}}[U]$')
    #ax1.plot(xi, np.abs(U), color='k', lw=1., dashes=[1,1], label=r'$|U|$')
    #ax1.axhline(0,color='k',lw=0.5)

    ax2.plot(iter_list, np.log10(acc_list), color='C0',lw=1.)


    # -------

    y_lim = (-0.4, 1.1)
    y_ticks = (-0.3,0.,0.3,0.6,0.9)
    ax1.tick_params(axis="y", length=2.0, pad=1)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_ylabel(r"Solution $U_0(\xi)$")

    x_lim = (-9,9)
    x_ticks = (-8,-4,0,4,8)
    #x_lim = (-13,13)
    #x_ticks = (-12,-6,0,6,12)
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax1.set_xlabel(r"Coordinate $\xi$", labelpad=1)

    legend = ax1.legend(
        ncol=1,
        handlelength=1.5,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=7.0,
        title_fontsize=6.0,
        labelcolor="k",
        loc = "upper right"
    )

    # -------

    x_lim = (0,35)
    x_ticks = (0,10,20,30)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax2.set_xlabel(r"Iteration $n$", labelpad=1)

    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-12.5,0.5))
    ax2.set_yticks((0,-2,-4,-6,-8,-10,-12))
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_{n})$")

    ax1.yaxis.set_label_coords(-0.16,0.5)
    ax2.yaxis.set_label_coords(-0.18,0.5)



def subfig_cd(fig, axs):
    ax1, ax2, ax3 = axs

    def _fetch_data(f_name):
        dat = np.load(f_name)
        eta = dat['eta']
        xi = dat['xi']
        U = dat['U']
        del_rle = dat['del']
        h = dat['h']
        return eta, xi, U, del_rle, h


    def set_colorbar_lin(fig, img, ax, label='text', dw=0.):
        # -- EXTRACT POSITION INFORMATION FOR COLORBAR PLACEMENT
        refPos = ax.get_position()
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        # -- SET NEW AXES AS REFERENCE FOR COLORBAR
        colorbar_axis = fig.add_axes([x0+0.79*w, y0 + .03*h, 0.2*w, 0.06*h])
        # -- SET CUSTOM COLORBAR
        colorbar = fig.colorbar(img,        # image described by colorbar
                cax = colorbar_axis,        # reference axex
                orientation = 'horizontal', # colorbar orientation
                extend = 'neither'             # ends with out-of range values
                )

        colorbar.outline.set_color('white')

        colorbar.ax.tick_params(
                color = 'white',                # tick color 
                labelcolor = 'white',           # label color
                bottom = False,             # no ticks at bottom
                labelbottom = False,        # no labels at bottom
                labeltop = True,            # labels on top
                top = True,                 # ticks on top
                direction = 'out',          # place ticks outside
                length = 2,                 # tick length in pts. 
                labelsize = 6.,             # tick font in pts.
                pad = 1.                    # tick-to-label distance in pts.
                )
        fig.text(x0 + 0.66*w, y0+0.03*h, label, horizontalalignment='left', verticalalignment='bottom', size=7, color='white')
        return colorbar

    # -- TOP SUBFIGURE

    eta, xi, U, del_rle, h = _fetch_data('../res.npz')

    eta_lim = (0,85)
    eta_ticks = (0,20,40,60,80)
    xi_lim = (5,-45)
    xi_ticks = (0,-10,-20,-30,-40)

    I = np.abs(np.swapaxes(U,0,1))**2
    img = ax1.pcolorfast(eta, xi, I[:-1,:-1],
                          vmin=0, vmax=np.max(I),
                          cmap = mpl.cm.get_cmap('turbo')
                          )

    cb = set_colorbar_lin(fig, img, ax1, label=r'$|\psi(\eta,\xi)|^2$')
    cb.set_ticks((0,1,2,3))

    ax1.tick_params(axis='y', length=2., pad=2, top=False)
    ax1.set_ylim(xi_lim)
    ax1.set_yticks(xi_ticks)
    ax1.set_ylabel(r"Coordinate $\xi$")

    ax1.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
    ax1.set_xlim(eta_lim)
    ax1.set_xticks(eta_ticks)
    #ax1.set_xlabel(r"Propagation distance $\eta$")


    # -- MIDDLE SUBFIGURE

    ax2.plot(eta, del_rle*1e8, color='C0', lw=1)
    ax2.axhspan(0.5,1.0,color='lightgray')


    y_lim = (0.25,1.25)
    y_ticks = (0.3,0.6,0.9,1.2)

    #y_lim = (0,1.6)
    #y_ticks = (0,0.5,1.,1.5)

    ax2.tick_params(axis='y', length=2., pad=2, top=False)
    ax2.set_ylim(y_lim)
    ax2.set_yticks(y_ticks)
    ax2.set_ylabel(r"RLE $\delta\times 10^8$")

    ax2.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
    ax2.set_xlim(eta_lim)
    ax2.set_xticks(eta_ticks)
    #ax2.set_xlabel(r"Propagation distance $\eta$")


    # -- BOTTOM SUBFIGURE

    ax3.plot(eta, h*1000, color='C0', lw=1)

    y_lim = (0,5)
    y_ticks = (0,2,4)

    #y_lim = (0,6.5)
    #y_ticks = (0,2,4,6)

    ax3.tick_params(axis='y', length=2., pad=2, top=False)
    ax3.set_ylim(y_lim)
    ax3.set_yticks(y_ticks)
    ax3.set_ylabel(r"Stepsize $h \times 10^3$")

    ax3.tick_params(axis='x', length=2., pad=2, top=False)
    ax3.set_xlim(eta_lim)
    ax3.set_xticks(eta_ticks)
    ax3.set_xlabel(r"Propagation distance $\eta$")


    ax1.yaxis.set_label_coords(-0.07,0.5)
    ax2.yaxis.set_label_coords(-0.07,0.5)
    ax3.yaxis.set_label_coords(-0.07,0.5)




main()
