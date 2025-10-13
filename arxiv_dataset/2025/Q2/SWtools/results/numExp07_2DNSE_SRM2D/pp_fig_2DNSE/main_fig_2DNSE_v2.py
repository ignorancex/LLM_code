import sys
import os
import scipy.optimize as so
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import axes3d



def save_fig(fig_name='test', fig_format='png'):
    #dir_name = os.path.dirname(fig_name)
    #os.makedirs(dir_name,exist_ok=True)
    if fig_format == 'png':
        plt.savefig(fig_name+'.png', format='png', dpi=600)
    elif fig_format == 'pdf':
        plt.savefig(fig_name+'.pdf', format='pdf', dpi=600)
    elif fig_format == 'svg':
        plt.savefig(fig_name+'.svg', format='svg', dpi=600)
    else:
        plt.show()


def set_style(fig_width=3.25, aspect_ratio = 0.6):

    #Axes3D.get_proj = get_proj_custom

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

    o_name = './fig_2DNSE_v2'
    o_format = 'png'

    def subfig_label(ax, label):
        pos = ax.get_position()
        fig.text(
            pos.x0,
            pos.y1,
            label,
            color="white",
            backgroundcolor="k",
            bbox=dict(facecolor="k", edgecolor="none", boxstyle="square,pad=0.1", linewidth=0),
            verticalalignment="top",
            horizontalalignment="left",
        )

    set_style(3.5, 0.66)
    fig = plt.figure()
    plt.subplots_adjust(left = 0.095, bottom = 0.11, right = 0.99, top = 0.98)
    gs00 = GridSpec(nrows = 1, ncols = 1)

    gsA = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs00[0,0], wspace=0.3, hspace=0.075)
    x0=0.01
    y0=0.08
    x1=0.55
    y1=1.08
    ax1 = fig.add_axes((x0,y0,x1-x0,y1-y0), projection='3d')
    ax1.set_box_aspect(aspect=(1, 1, 1.8))

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
        acc_list = dat['acc_list']
        U = dat['U']
        xi = dat['xi']
        return xi, U, iter_list, acc_list

    # -------

    xi, U, iter_list, acc_list = _fetch_data('../res_2DNSE_SRM2D_kap3.704500.npz')

    # -- LEFT SUBPLOT
    xx, yy = xi

    xMask = np.abs(xx[:,0])<7
    yMask = np.abs(yy[0,:])<7
    def _trim(u):
        u = u[xMask,:]
        u=u[:,yMask]
        return u


    myCmap = mpl.cm.get_cmap('turbo')

    ax1.plot_surface(_trim(xx), _trim(yy), _trim(np.abs(U)), edgecolor='k', lw=0.4, rstride=3, cstride=3, alpha=.9, cmap=myCmap)
    ##ax1.plot_surface(_trim(xx), _trim(yy), _trim(np.abs(U)), edgecolor='C0', lw=0.25, rstride=1, cstride=1, alpha=0.3)

    # -- RIGHT SUBPLOT
    ax2.plot(iter_list, np.log10(acc_list), color='C0', lw=1.)

    # -------

    y_lim = (-7,7)
    y_ticks = ((-6,-3,0,3,6))
    ax1.tick_params(axis="y", length=2.0, pad=-5)
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    ax1.set_ylabel(r'Coordinate $\xi_2$',labelpad=-10)

    x_lim = (-7,7)
    x_ticks = ((-6,-3,0,3,6))
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis="x", length=2.0, pad=-5)
    ax1.set_xlabel(r'Coordinate $\xi_1$',labelpad=-10)

    z_lim = (0,1.)
    z_ticks = ((0,0.2,0.4,0.6,0.8,1.0))
    ax1.set_zlim(z_lim)
    ax1.set_zticks(z_ticks)
    ax1.tick_params(axis="z", length=2.0, pad=-2)
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel(r"Solution $U(\xi)$",labelpad=-6,rotation=90)

    ax1.view_init(25, -135)

    # -------

    x_lim = (0,3200)
    x_ticks = (0,1000,2000,3000)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis="x", length=2.0, pad=1, top=False, direction='out')
    ax2.set_xlabel(r"Iteration $n$", labelpad=1)

    ax2.tick_params(axis="y", length=2.0, pad=1, labelleft=True, direction='out')
    ax2.set_ylim((-12.5,-1.5))
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_{n})$")

    ax1.yaxis.set_label_coords(-0.16,0.5)
    ax2.yaxis.set_label_coords(-0.18,0.5)






main()
