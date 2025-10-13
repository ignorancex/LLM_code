"""main_fig_stability_v1.py

Module implementing a figure with the description:

Linear stability analysis for the solitary waves of a higher-order nonlinear
Schrödinger equation with quartic dispersion.
%
(a) Solitary wave with wavenumber $\kappa$.
%
(b) Zero-eigenvalue modes (stars) and internal modes (diamonds).
Continuous-wave spectrum is shaded gray.
%
(c) Small-amplitude mode $f$ of the fundamental internal mode, and,
%
(d) corresponding mode $g$.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def main_figure(res, o_name = './fig_v1'):

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
    plt.subplots_adjust(left = 0.1, bottom = 0.11, right = 0.99, top = 0.98, wspace=1.3)
    gs00 = GridSpec(nrows = 1, ncols = 1)

    gsA = GridSpecFromSubplotSpec(4, 5, subplot_spec=gs00[0,0], wspace=1.5, hspace=0.2)
    ax0 = fig.add_subplot(gsA[:, 2])

    gsB = GridSpecFromSubplotSpec(2, 1, subplot_spec=gsA[:,3:], wspace=0.075, hspace=0.05)
    ax1 = fig.add_subplot(gsB[0, 0])
    ax2 = fig.add_subplot(gsB[1, 0])
    sf1 = [ax1,ax2]

    gsC = GridSpecFromSubplotSpec(1, 1, subplot_spec=gsA[:,:2], wspace=0.075, hspace=0.075)
    ax3 = fig.add_subplot(gsC[0, 0])


    # -- SUBFIGURE CONTENT ----------------------------------------------------
    subfig_a(fig, ax3, res)
    subfig_b(fig, ax0, res)
    subfig_cd(fig, sf1, res)

    subfig_label(ax3,r"(a)")
    subfig_label(ax0,r"(b)")
    subfig_label(ax1,r"(c)")
    subfig_label(ax2,r"(d)")

    # -- GENERATE FIGURE ------------------------------------------------------
    save_fig(fig_name=o_name, fig_format=o_format )


def marker_style(LamR, LamI):

    my_col, my_marker, my_size = 'k', 'o', 4
    mfc, mew = 'k',1

    if np.abs(LamR) < 0.02 and np.abs(LamI) < 0.01:
        my_col, my_marker, my_size = 'limegreen', '*', 6
        mfc, mew = 'limegreen', 0
    elif np.abs(LamR) > 0.02 and np.abs(LamI) < 0.01:
        my_col, my_marker, my_size = 'magenta', 'd', 4
        mfc, mew = 'white', 1
    elif np.abs(LamI) > 0.01:
        my_col, my_marker, my_size = 'C0', 'd', 4
        mfc, mew = 'C0', 1

    return my_col, my_marker, my_size, mfc, mew


def subfig_a(fig, ax, res):

    t = res['xi']
    U = res['U']
    kap = res['kap']

    def my_label(ax, label):
        pos = ax.get_position()
        fig.text(
            pos.x0+0.01,
            pos.y0+0.01,
            label,
            color="k",
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    my_yLims = lambda y: (1.2*np.min(np.real(y)),1.2*np.max(np.real(y)))

    ax.axhline(0, color='k', lw=0.75)
    ax.plot(t, np.real(U), color="C0", lw=1, label=r'${\mathrm{Re}}[U]$')
    ax.plot(t, np.imag(U), color="C0", dashes=[2,1], lw=1, label=r'${\mathrm{Im}}[U]$')
    ax.set_ylim(my_yLims(U))
    ax.set_ylabel(r"Solution $U(\xi)$")
    ax.tick_params(axis="x", length=2.0, pad=1)
    ax.tick_params(axis="y", length=2.0, pad=1)
    ax.set_xlabel(r"Coordinate $\xi$", labelpad=1)

    ax.legend(
        ncol=1,
        loc='upper right',
        handlelength=0.8,
        columnspacing=1.,
        handletextpad=0.5
    )

    #my_label(ax, "$\kappa = %4.3lf$"%(kap))
    ax.yaxis.set_label_coords(-0.25,0.5)


def subfig_b(fig, ax, res):
    kap = res['kap']
    Lam = res['Lam']
    Lam_max = res['Lam_max']
    f = res['f']
    g = res['g']

    ax.axhline(0,color='k', lw=0.5)
    ax.axvline(0,color='k', lw=0.5)
    for i in range(Lam.size):
        LamR, LamI = np.real(Lam[i]), np.imag(Lam[i])
        my_col, my_marker, my_size, mfc, mew = marker_style(LamR, LamI)
        ax.plot([LamR], [LamI], color=my_col, marker = my_marker, markersize=my_size, mfc=mfc, mew=mew)

    ax.axhline(Lam_max, color='k', dashes=[1,1])
    ax.axhline(-Lam_max, color='k', dashes=[1,1])

    ax.axhspan(Lam_max,1.5*Lam_max, color='silver')
    ax.axhspan(-Lam_max,-1.5*Lam_max, color='silver')

    y_lim = (-1.2*Lam_max, 1.2*Lam_max)
    ax.tick_params(axis="y", length=2.0, pad=1)
    ax.set_ylim(y_lim)
    ax.set_ylabel(r"$\rm{Im}[\Lambda]$")

    x_lim = (-1.5,1.5)
    ax.set_xlim(x_lim)
    ax.tick_params(axis="x", length=2.0, pad=1, top=False)
    ax.set_xlabel(r"$\rm{Re}[\Lambda]$", labelpad=1)

    ax.yaxis.set_label_coords(-.9,0.5)


def subfig_cd(fig, axs, res):

    ax1, ax2 = axs

    t = res['xi']
    Lam = res['Lam']
    f_ = res['f']
    g_ = res['g']

    my_yLims = lambda y: (1.2*np.min(np.real(y)),1.4*np.max(np.real(y)))

    idx = np.argmin(np.where(np.abs(np.imag(Lam))>0.02,  np.abs(np.imag(Lam)) , np.inf))

    LamR, LamI = np.real(Lam[idx]), np.imag(Lam[idx])
    f, g = f_[idx], g_[idx]

    my_col, my_marker, my_size, mfc, mew = marker_style(LamR, LamI)

    ax1.axhline(0, color='k', lw=0.75)
    ax1.plot(t, np.real(f), color=my_col, lw=1, label=r'${\mathrm{Re}}[f]$')
    ax1.plot(t, np.imag(f), color=my_col, dashes=[2,1], lw=1, label=r'${\mathrm{Im}}[f]$')
    ax1.set_ylim(my_yLims(f))
    ax1.set_ylabel(r"Mode $f$")
    #ax1.set_ylabel(r"Mode $f$ (${\rm{Im}}[\Lambda] = %4.3lf$)"%(LamI))
    ax1.tick_params(axis="x", length=2.0, pad=1, labelbottom=False)
    ax1.tick_params(axis="y", length=2.0, pad=1)

    ax1.legend(
        ncol=2,
        handlelength=0.8,
        columnspacing=0.75,
        handletextpad=0.5,
        loc = (0.15,0.85)
    )

    ax2.axhline(0, color='k', lw=0.75)
    ax2.plot(t, np.real(g), color=my_col, lw=1, label=r'${\mathrm{Re}}[g]$')
    ax2.plot(t, np.imag(g), color=my_col, dashes=[2,1], lw=1, label=r'${\mathrm{Im}}[g]$')
    ax2.set_ylim(my_yLims(g))
    ax2.set_ylabel(r"Mode $g$")
    #ax2.set_ylabel(r"Mode $g$ (${\rm{Im}}[\Lambda] = %4.3lf$)"%(LamI))
    ax2.tick_params(axis="x", length=2.0, pad=1)
    ax2.tick_params(axis="y", length=2.0, pad=1)
    ax2.set_xlabel(r"Coordinate $\xi$", labelpad=1)

    ax2.legend(
        ncol=2,
        handlelength=0.8,
        columnspacing=0.75,
        handletextpad=0.5,
        loc = (0.15,0.85)
    )

    ax1.yaxis.set_label_coords(-0.3,0.5)
    ax2.yaxis.set_label_coords(-0.3,0.5)


if __name__=="__main__":
    res = np.load('../res_LE_HONSE.npz')
    main_figure(res, o_name = './fig_HONSE_stability')
