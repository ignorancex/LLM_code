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

    o_name = './fig_verification_v2'
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
    plt.subplots_adjust(left = 0.125, bottom = 0.11, right = 0.99, top = 0.98)
    gs00 = GridSpec(nrows = 1, ncols = 1)

    gsA = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs00[0,0], wspace=0.3, hspace=0.075)
    ax1 = fig.add_subplot(gsA[0, 0])
    ax2 = fig.add_subplot(gsA[0, 1])
    sf1=[ax1, ax2]

    ax1 = fig.add_subplot(gsA[1, 0])
    ax2 = fig.add_subplot(gsA[1, 1])
    sf2=[ax1, ax2]



    # -- SUBFIGURE CONTENT ----------------------------------------------------
    subfig_ab(fig, sf1)
    subfig_label(sf1[0],r"(a)")
    subfig_label(sf1[1],r"(b)")

    subfig_cd(fig, sf2)
    subfig_label(sf2[0],r"(c)")
    subfig_label(sf2[1],r"(d)")

    # -- GENERATE FIGURE ------------------------------------------------------
    save_fig(fig_name=o_name, fig_format=o_format )


def subfig_ab(fig, axs):
    ax1, ax2 = axs

    def _fetch_data(f_name):
        dat = np.loadtxt(f_name)
        N = dat[:,0]
        dt = dat[:,1]
        K = dat[:,4]
        err = dat[:,6]
        return dt, K, err

    # -------


    dt, K, err = _fetch_data('../res_NSOM_ORP1.0.dat')
    ax1.plot(dt, K, color='C0', dashes=[], marker='o', lw=1., markersize=2.5, label=r'1.0')
    ax2.plot(dt, np.log10(err), color='C0', dashes=[], marker='o', lw=1., markersize=2.5, label=r'1.0')

    ax1.axhline(0.96, color='k', dashes=[2,1], lw=0.75)

    dt_ = np.linspace(0.07,0.2,100)
    ax2.plot(dt_, np.log10(0.04*dt_**2), color='k', dashes=[], lw=0.75)



    # -------

    y_lim = (0.9595, 0.9665)
    y_ticks = (0.960,0.962,0.964,0.966)
    ax1.tick_params(axis="y", length=2.0, pad=1)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(("$\kappa$","0.962","0.964","0.966"))
    ax1.set_ylabel(r"Eigenvalue ${\rm{K}}^\star$")

    #x_lim = (0.01,0.5)
    x_lim = (0.015,0.4)
    x_ticks = (0.02,0.04,0.08,0.16,0.32)
    ax1.set_xscale('log')
    ax1.tick_params(axis="x", length=2.0, pad=1, top=False)
    #ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.minorticks_off()
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)


    legend = ax1.legend(
        ncol=1,
        title=r'ORP $\omega$',
        handlelength=1.85,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=6.0,
        title_fontsize=6.0,
        labelcolor="k",
        loc = "upper center"
        #loc = "center left"
        #loc = (0.72,0.125)
        #loc = (0.05,0.45)
    )

    # -------

    ax2.set_xscale('log')
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.minorticks_off()
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)

    #ax2.axes.set_yscale('log')
    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-4.5,-0.8))
    ax2.set_yticks((-1,-2,-3,-4))
    #ax2.set_ylim((5e-5,0.03))
    #ax2.set_yticks((1e-2,1e-3,1e-4))
    ax2.set_ylabel(r"log-error $\log(\epsilon)$")
    #ax2.set_ylabel(r"RMS error $\epsilon_{\rm{glob}}$")

    pos = ax2.get_position()
    x_pos = pos.x0 + 0.62*(pos.x1-pos.x0)
    y_pos = pos.y0 + 0.36*(pos.y1-pos.y0)
    fig.text(
        x_pos,
        y_pos,
        r"$\propto \Delta \xi^2$",
        color="k",
        verticalalignment="top",
        horizontalalignment="left",
    )


    legend = ax2.legend(
        ncol=1,
        title=r'ORP $\omega$',
        handlelength=1.85,
        borderpad=0.1,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.2,
        fontsize=6.0,
        title_fontsize=6.0,
        labelcolor="k",
        loc="upper center"
        #loc="upper right"
    )

    def _subfig_label(ax, label):
        pos = ax.get_position()
        x_pos = pos.x0 + 0.970*(pos.x1-pos.x0)
        y_pos = pos.y0 + 0.960*(pos.y1-pos.y0)
        fig.text(
            x_pos,
            y_pos,
            label,
            color="k",
            verticalalignment="top",
            horizontalalignment="right",
        )

    _subfig_label(ax1, "NSOM")
    _subfig_label(ax2, "NSOM")

    ax1.yaxis.set_label_coords(-0.24,0.5)
    ax2.yaxis.set_label_coords(-0.19,0.5)



def subfig_cd(fig, axs):
    ax1, ax2 = axs

    def _fetch_data(f_name):
        dat = np.loadtxt(f_name)
        N = dat[:,0]
        dt = dat[:,1]
        E = dat[:,2]
        err = dat[:,6]
        return dt, E, err

    # -------

    dt, E, err = _fetch_data('../res_SRM.dat')
    ax1.plot(dt, E, color='C0', marker='o', lw=1., markersize=2.5)
    ax1.axhline(3.0983866769630657, color='k', dashes=[2,1], lw=0.75)

    ax2.plot(dt, np.log10(err), color='C0', marker='o', lw=1., markersize=2.5)

    # -------

    y_lim = (3.09675,3.10025)
    y_ticks = (3.097,3.098,3.0983867,3.099,3.100)
    #y_lim = (3.098175,3.098625)
    #y_ticks = (3.0982,3.0983,3.0984,3.0985,3.0986)
    ax1.tick_params(axis="y", length=2.0, pad=1)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(("3.097","3.098","$N$","3.099","3.100"))
    ax1.set_ylabel(r"Energy $N^\star$")

    x_lim = (0.015,0.4)
    x_ticks = (0.02,0.04,0.08,0.16,0.32)
    ax1.set_xscale('log')
    ax1.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.minorticks_off()
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks)
    ax1.set_xlabel(r"Mesh size $\Delta \xi$", labelpad=1)


    legend = ax1.legend(
        ncol=1,
        #title=r'$\kappa$',
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

    ax2.set_xscale('log')
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False, bottom=True)
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.minorticks_off()
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticks)
    ax2.set_xlabel(r"Mesh size $\Delta \xi$",labelpad=1)

    #ax2.axes.set_yscale('log')
    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True)
    ax2.set_ylim((-13.5,-6.5))
    ax2.set_yticks((-7,-9,-11,-13))
    #ax2.set_ylim((0.5e-14,2e-6))
    #ax2.set_yticks((1e-14,1e-12, 1e-10, 1e-8, 1e-6))
    ax2.set_ylabel(r"log-error $\log(\epsilon)$")


    legend = ax2.legend(
        ncol=1,
        #title=r'$\kappa$',
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


    def _subfig_label(ax, label):
        pos = ax.get_position()
        x_pos = pos.x0 + 0.970*(pos.x1-pos.x0)
        y_pos = pos.y0 + 0.960*(pos.y1-pos.y0)
        fig.text(
            x_pos,
            y_pos,
            label,
            color="k",
            verticalalignment="top",
            horizontalalignment="right",
        )

    _subfig_label(ax1, "SRM")
    _subfig_label(ax2, "SRM")

    ax1.yaxis.set_label_coords(-0.24,0.5)
    ax2.yaxis.set_label_coords(-0.19,0.5)



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
