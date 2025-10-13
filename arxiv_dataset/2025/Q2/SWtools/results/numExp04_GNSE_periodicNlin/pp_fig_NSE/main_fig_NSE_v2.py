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

    o_name = './fig_1DNSE_nlinMicro_v2'
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
    plt.subplots_adjust(left = 0.095, bottom = 0.11, right = 0.99, top = 0.98)
    gs00 = GridSpec(nrows = 1, ncols = 1)

    gsA = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs00[0,0], wspace=0.45, hspace=0.075)
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
    ax1_inset = ax1.inset_axes([0.63,0.785, 0.35,0.2])
    ax3 = ax1.twinx()

    def _fetch_data(f_name):
        dat = np.load(f_name)
        iter_list = dat['iter_list']
        kap_list = dat['kap_list']
        U_list = dat['U_list']
        err_list = dat['acc_list']
        U = dat['U']
        t = dat['t']
        kap = dat['kap']
        alpha = dat['alpha']
        return t, U, alpha, iter_list, kap_list, U_list, err_list

    # -------

    x, U, alpha, iter_list, mu_list, U_list, err_list = _fetch_data('../data_GNSE/res_GNSE_SRM_kap1.000000_alpha-0.300000.npz')
    ax1.plot(x, U, color='C0', dashes=[], lw=1., label=r"$%2.1lf$"%(alpha))
    ax1_inset.plot(x, U, color='C0', dashes=[], lw=1., label=r"$%2.1lf$"%(alpha))
    ax2.plot(iter_list, np.log10(err_list), color='C0', dashes=[], lw=1., label="$%2.1lf$"%(alpha))
    m = alpha*np.cos(4*np.pi*x)
    ax3.plot(x, m/np.abs(alpha), color='silver', dashes=[], lw=1.)

    x, U, alpha, iter_list, mu_list, U_list, err_list = _fetch_data('../data_GNSE/res_GNSE_SRM_kap1.000000_alpha-0.800000.npz')
    ax1.plot(x, U, color='C0', dashes=[3,1,1,1], lw=1., label=r"$%2.1lf$"%(alpha))
    ax1_inset.plot(x, U, color='C0', dashes=[3,1,1,1], lw=1., label=r"$%2.1lf$"%(alpha))
    ax2.plot(iter_list, np.log10(err_list), color='C0', dashes=[3,1,1,1], lw=1., label="$%2.1lf$"%(alpha))

    x, U, alpha, iter_list, mu_list, U_list, err_list = _fetch_data('../data_GNSE/res_GNSE_SRM_kap1.000000_alpha-3.000000.npz')
    ax1.plot(x, U, color='C0', dashes=[2,1], lw=1., label=r"$%2.1lf$"%(alpha))
    ax1_inset.plot(x, U, color='C0', dashes=[2,1], lw=1., label=r"$%2.1lf$"%(alpha))
    ax2.plot(iter_list, np.log10(err_list), color='C0', dashes=[2,1], lw=1., label="$%2.1lf$"%(alpha))

    x, U, alpha, iter_list, mu_list, U_list, err_list = _fetch_data('../data_GNSE/res_GNSE_SRM_kap1.000000_alpha-8.000000.npz')
    ax1.plot(x, U, color='C0', dashes=[1,1], lw=1., label=r"$%2.1lf$"%(alpha))
    ax1_inset.plot(x, U, color='C0', dashes=[1,1], lw=1., label=r"$%2.1lf$"%(alpha))
    ax2.plot(iter_list, np.log10(err_list), color='C0', dashes=[1,1], lw=1., label="$%2.1lf$"%(alpha))

    # -------

    y_lim = (0, 1.85)
    y_ticks = (0,0.4,0.8,1.2,1.6)
    ax1.tick_params(axis="y", length=2.0, pad=1)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_ylabel(r"Solution $U(\xi)$")

    x_lim = (-4.5,4.5)
    x_ticks = (-4,-2,0,2,4)
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax1.set_xlabel(r"Coordinate $\xi$", labelpad=1)

    legend = ax1.legend(
        ncol=1,
        title=r'$\alpha$',
        handlelength=1.85,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=6.0,
        title_fontsize=6.0,
        labelcolor="k",
        #loc = "upper left"
        #loc = "upper right"
        #loc = (0.7,0.45)
        loc = (0.15,0.765)
    )


    # -------



    y_lim = (1.375, 1.415)
    y_ticks = (1.38,1.39,1.4,1.41)
    #y_lim = (1.18, 1.42)
    #y_ticks = (1.2,1.3,1.4)
    ax1_inset.tick_params(axis="y", length=2.0, pad=1, labelsize=5)
    ax1_inset.set_ylim(y_lim)
    ax1_inset.set_yticks(y_ticks)

    x_lim = (-0.25,0.25)
    x_ticks = (-0.2,0,0.2)
    ax1_inset.tick_params(axis="x", length=2.0, pad=1, labelsize=5)
    ax1_inset.set_xlim(x_lim)
    ax1_inset.set_xticks(x_ticks)

    my_col = 'k'
    ax1_inset.spines['top'].set_color(my_col)
    ax1_inset.spines['bottom'].set_color(my_col)
    ax1_inset.spines['left'].set_color(my_col)
    ax1_inset.spines['right'].set_color(my_col)

    x_min, x_max = x_lim
    y_min, y_max = y_lim
    box_x = [x_min, x_min, x_max, x_max, x_min]
    box_y = [y_min, y_max, y_max, y_min, y_min]
    ax1.plot(box_x, box_y, color=my_col, lw=0.75)

    # -------

    ax3.tick_params(axis="y", length=2.0, pad=1)
    ax3.set_ylim(-1,8)
    ax3.set_yticks((-1,0,1))
    ax3.set_ylabel(r"$m(\xi)/|\alpha|$")
    ax3.yaxis.set_label_coords(1.1,0.12)

    # -------

    x_lim = (0,45)
    x_ticks = (0,10,20,30,40)
    #x_lim = (0,4300)
    #x_ticks = (0,1000,2000,3000,4000)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax2.set_xlabel(r"Iteration $n$", labelpad=1)

    #ax2.axes.set_yscale('log')
    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-12.5,0.5))
    ax2.set_yticks((-12,-10,-8,-6,-4,-2,0))
    #ax2.set_ylim((1e-12,1.))
    #ax2.set_yticks((1e0,1e-2,1e-4,1e-6,1e-8, 1e-10, 1e-12))
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_{n})$")

    legend = ax2.legend(
        ncol=1,
        title=r'$\alpha$',
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

    ax1.yaxis.set_label_coords(-0.16,0.5)
    ax2.yaxis.set_label_coords(-0.18,0.5)


main()
