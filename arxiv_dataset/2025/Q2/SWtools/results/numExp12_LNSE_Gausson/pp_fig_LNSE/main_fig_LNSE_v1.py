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

    o_name = './fig_LNSE_v1'
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

    set_style(3.5, 0.66)
    fig = plt.figure()
    plt.subplots_adjust(left = 0.11, bottom = 0.11, right = 0.99, top = 0.98)
    gs00 = GridSpec(nrows = 1, ncols = 1)

    gsA = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs00[0,0], wspace=0.3, hspace=0.075)
    ax1 = fig.add_subplot(gsA[0, 0])
    ax2 = fig.add_subplot(gsA[0, 1])
    sf1=[ax1, ax2]


    # -- SUBFIGURE CONTENT ----------------------------------------------------
    subfig_ab(fig, sf1)

    subfig_label(sf1[0],r"(a)")
    subfig_label(sf1[1],r"(b)")

    # -- GENERATE FIGURE ------------------------------------------------------
    save_fig(fig_name=o_name, fig_format=o_format )



def subfig_ab(fig, axs):
    ax1, ax2 = axs

    def _fetch_data(f_name):
        dat = np.load(f_name)
        iter_list = dat['iter_list']
        kap_list = dat['kap_list']
        U_list = dat['U_list']
        err_list = dat['acc_list']
        U = dat['U']
        xi = dat['xi']
        kap = dat['kap']

        xi0 = np.sum(xi*np.abs(U)**2)/np.sum(np.abs(U)**2)

        return xi-xi0, U, kap, iter_list, kap_list, U_list, err_list

    # -------

    f_name = sys.argv[1]
    xi, U, kap, iter_list, mu_list, U_list, err_list = _fetch_data(f_name)
    ax1.plot(xi, np.abs(U), color='C0', dashes=[], lw=1., label=r"$U$")
    ax2.plot(iter_list, np.log10(err_list), color='C0', dashes=[], lw=1., label="$U$")

    UG = np.exp(-xi**2/2)/np.pi**0.25
    ax1.plot(xi, UG, color='C3', dashes=[3,2], lw=1.3, label=r"$U_G$")

    # -------

    y_lim = (1e-21, 4)
    y_ticks = (1e-20, 1e-15, 1e-10, 1e-5,1)
    ax1.axes.set_yscale('log')
    ax1.tick_params(axis="y", length=2.0, pad=1)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_ylabel(r"Solution $U(\xi)$")

    x_lim = (-11,11)
    x_ticks = (-10,-5,0,5,10)
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax1.set_xlabel(r"Coordinate $\xi$", labelpad=1)

    legend = ax1.legend(
        ncol=1,
        handlelength=1.85,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=6.0,
        title_fontsize=6.0,
        labelcolor="k",
        loc = "upper right"
    )

    # -------

    x_lim = (0,2200)
    x_ticks = (0,500,1000,1500,2000)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax2.set_xlabel(r"Iteration $n$", labelpad=1)

    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-12.5,.5))
    ax2.set_yticks((-12,-10,-8,-6,-4,-2,0))
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_{n})$")

    legend = ax2.legend(
        ncol=1,
        handlelength=1.85,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=6.0,
        title_fontsize=6.0,
        labelcolor="k",
        loc="upper right"
    )

    ax1.yaxis.set_label_coords(-0.2,0.5)
    ax2.yaxis.set_label_coords(-0.18,0.5)


main()
