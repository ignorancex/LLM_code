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

    o_name = './fig_1DGPE_v2'
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
        iter_list = dat['iter_list']
        kap_list = dat['kap_list']
        U_list = dat['U_list']
        err_list = dat['acc_list']
        U = dat['U']
        x = dat['x']
        beta = dat['beta']
        # -- CALCULATE ENERGY AS FUNTION OF ITERATION 
        E_list = -kap_list + 0.5*beta*np.trapz(np.abs(U_list)**4,x=x, axis=-1)
        # -- CENTER POSITION
        xc = np.trapz(x*np.abs(U)**2,x=x)/np.trapz(np.abs(U)**2,x=x)
        print(f"beta = {beta}, xc = {xc}")
        return x, U, beta, iter_list, -kap_list, U_list, err_list # E_list/E_list[-1] - 1

    # -------

    x, U, beta, iter_list, mu_list, U_list, err_list = _fetch_data('../data_1DGPE/res_1DGPE_SOR_beta3.137100_wORP1.500000.npz')
    ax1.plot(x, U, color='C0', lw=1., label=r"$%5.4lf$"%(beta))
    ax2.plot(iter_list, np.log10(err_list), color='C0', lw=1., label="$%5.4lf$"%(beta))

    x, U, beta, iter_list, mu_list, U_list, err_list = _fetch_data('../data_1DGPE/res_1DGPE_SOR_beta31.371000_wORP1.500000.npz')
    ax1.plot(x, U, color='C0', dashes=[3,1,1,1], lw=1., label=r"$%5.3lf$"%(beta))
    ax2.plot(iter_list, np.log10(err_list), color='C0', dashes=[3,1,1,1], lw=1., label="$%5.3lf$"%(beta))

    x, U, beta, iter_list, mu_list, U_list, err_list = _fetch_data('../data_1DGPE/res_1DGPE_SOR_beta156.855000_wORP1.500000.npz')
    ax1.plot(x, U, color='C0', dashes=[2,1], lw=1., label=r"$%5.2lf$"%(beta))
    ax2.plot(iter_list, np.log10(err_list), dashes=[2,1], color='C0', lw=1., label="$%5.2lf$"%(beta))

    x, U, beta, iter_list, mu_list, U_list, err_list = _fetch_data('../data_1DGPE/res_1DGPE_SOR_beta627.420000_wORP1.500000.npz')
    ax1.plot(x, U, color='C0', dashes=[1,1], lw=1., label=r"$%5.2lf$"%(beta))
    ax2.plot(iter_list, np.log10(err_list), dashes=[1,1], color='C0', lw=1., label="$%5.2lf$"%(beta))

    # -------

    y_lim = (0, 0.7)
    y_ticks = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7)
    ax1.tick_params(axis="y", length=2.0, pad=1)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_ylabel(r"Solution $U(\xi)$")

    x_lim = (0,12.5)
    x_ticks = (0,2,4,6,8,10,12)
    x_lim = (-13,13)
    x_ticks = (-12,-6,0,6,12)
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax1.set_xlabel(r"Coordinate $\xi$", labelpad=1)

    legend = ax1.legend(
        ncol=1,
        title=r'$\beta$',
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
    #x_lim = (0,4300)
    #x_ticks = (0,1000,2000,3000,4000)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax2.set_xlabel(r"Iteration $n$", labelpad=1)

    #ax2.axes.set_yscale('log')
    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-12.5,-1.5))
    #ax2.set_yticks((1e-4,1e-6,1e-8, 1e-10, 1e-12))
    #ax2.set_ylim((1e-12,2e-3))
    #ax2.set_yticks((1e-4,1e-6,1e-8, 1e-10, 1e-12))
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_{n})$")

    legend = ax2.legend(
        ncol=1,
        title=r'$\beta$',
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
