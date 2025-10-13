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

    o_name = './fig_1DGPE_excitedStates_v2'
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
        x = dat['x']
        nskip = dat['nskip']
        U0_0 = dat['U0_0']
        U0_1 = dat['U0_1']
        U0_2 = dat['U0_2']
        U0_3 = dat['U0_3']
        U0 = dat['U0']
        kap0 = dat['kap0']
        err0 = dat['acc0']
        U1 = dat['U1']
        kap1 = dat['kap1']
        err1 = dat['acc1']
        U2 = dat['U2']
        kap2 = dat['kap2']
        err2 = dat['acc2']
        U3 = dat['U3']
        kap3 = dat['kap3']
        err3 = dat['acc3']
        # -- CENTER POSITION
        return x, U0, -kap0, err0, U1, -kap1, err1,   U2, -kap2, err2,   U3, -kap3, err3, U0_0, U0_1, U0_2, U0_3, nskip

    # -------

    _scale = lambda u: 0.3*u/np.max(np.abs(u))

    #x, U0, kap0, err0, U1, kap1, err1,   U2, kap2, err2,   U3, kap3, err3, U0_0, U0_1, U0_2, U0_3, nskip = _fetch_data('../data_1DGPE/res_1DGPE_ITP_beta0.500000.npz')
    x, U0, kap0, err0, U1, kap1, err1,   U2, kap2, err2,   U3, kap3, err3, U0_0, U0_1, U0_2, U0_3, nskip = _fetch_data('../data_1DGPE/res_1DGPE_NSOM_beta1.000000.npz')

    ax1.plot(x, 0.5*x**2, color='silver', dashes=[], lw=1.0, zorder=-1)

    ax1.fill_between(x,kap0, kap0 + _scale(U0), color='C0', alpha=0.5, lw=0., label=r'$U_0$')
    ax1.axhline(kap0, color='C0', dashes=[], lw=1., label=r'$\mu_0$')
    ax2.plot(nskip*np.arange(err0.size), np.log10(err0), color='C0', lw=1., label="$U_0$")

    ax1.fill_between(x,kap1, kap1 + _scale(U1), color='C1', alpha=0.5, lw=0., label=r'$U_1$')
    ax1.axhline(kap1, color='C1', dashes=[3,1,1,1], lw=1., label=r'$\mu_1$')
    ax2.plot(nskip*np.arange(err1.size), np.log10(err1), color='C1', dashes=[3,1,1,1], lw=1., label="$U_1$")

    ax1.fill_between(x,kap2, kap2 + _scale(U2), color='C2', alpha=0.5, lw=0., label=r'$U_2$')
    ax1.axhline(kap2, color='C2', dashes=[2,1], lw=1., label=r"$\mu_2$")
    ax2.plot(nskip*np.arange(err2.size), np.log10(err2), dashes=[2,1], color='C2', lw=1., label="$U_2$")

    ax1.fill_between(x,kap3, kap3 + _scale(U3), color='C3', alpha=0.5, lw=0., label=r'$U_3$')
    ax1.axhline(kap3, color='C3', dashes=[1,1], lw=1., label=r"$\mu_3$")
    ax2.plot(nskip*np.arange(err3.size), np.log10(err3), dashes=[1,1], color='C3', lw=1., label="$U_3$")

    # -------

    y_lim = (0, 5.2)
    y_ticks = (0.,0.5,1.5,2.5,3.5)
    #y_ticks = (0,0.5,1,1.5,2, 2.5, 3., 3.5, 4., 4.5, 5.)
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
        ncol=2,
        #title=r'$N$',
        handlelength=1.85,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=6.0,
        title_fontsize=6.0,
        labelcolor="k",
        loc = "upper center"
    )

    # -------

    x_lim = (0,3200)
    x_ticks = (0,1000,2000,3000)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax2.set_xlabel(r"Iteration $n$", labelpad=1)

    #ax2.axes.set_yscale('log')
    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-12.5,-3.5))
    ax2.set_yticks((-4,-6,-8,-10,-12))
    #ax2.set_ylim((1e-12,3e-4))
    #ax2.set_yticks((1e-4,1e-6,1e-8, 1e-10, 1e-12))
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_{n})$")

    legend = ax2.legend(
        ncol=1,
        #title=r'$N$',
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


def main_show():

    def _fetch_data(f_name):
        dat = np.load(f_name)
        kap = dat['res_kap']
        psi = dat['res_psi']
        t = dat['t']
        return t, psi[-1], kap[-1]

    t, psi, kap = _fetch_data('../data/res_NSE_GSrelax_wORP1.000000.npz')

    for i in range(t.size):
        print(t[i], psi[i])


main()
#main_show()
