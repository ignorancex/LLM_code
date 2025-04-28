from MESAreader import getSrcCol, Rsun_cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def get_Fe_core_mass(pfile, si_threshold=0.25):
    """we recalculate from the final profile the Fe core mass. The
    value in the MESA history file is too large and sentive to
    resolution because we redefine it to quench velocities only in the
    outer layers.
    """
    src, col = getSrcCol(pfile)
    si28 = src[:, col.index("si28")]
    m = src[:, col.index("mass")]
    return m[np.argmin(np.absolute(si28-si_threshold))]


def plot_ye(pfile, ax, **kwargs):
    src, col = getSrcCol(pfile)
    ye = src[:, col.index("ye")]
    m = src[:, col.index("mass")]
    ax.plot(m, ye, **kwargs)
    # logR = np.log10((10.0 ** src[:, col.index("logR")]) * Rsun_cm)
    # ax.plot(logR, ye, **kwargs)


def plot_j(pfile, ax, **kwargs):
    src, col = getSrcCol(pfile)
    j_specific = src[:, col.index("j_rot")]
    m = src[:, col.index("mass")]
    ax.plot(m, np.log10(j_specific), **kwargs)
    # logR = np.log10((10.0 ** src[:, col.index("logR")]) * Rsun_cm)
    # ax.plot(logR, np.log10(j_specific), **kwargs)

def plot_rho(pfile, ax, **kwargs):
    src, col = getSrcCol(pfile)
    logrho = src[:, col.index("logRho")]
    m = src[:, col.index("mass")]
    ax.plot(m, logrho, **kwargs)
    # logR = np.log10((10.0 ** src[:, col.index("logR")]) * Rsun_cm)
    # ax.plot(logR, logrho, **kwargs)


def compare_models_mass_coord(pfile1_low_res, pfile1_high_res,
                              pfile2_low_res=None, pfile2_high_res=None, fig_name=None):
    c1 = 'b'
    c2 = 'orange'
    ls_low_res = '--'
    ls_high_res = '-'
    lw_low_res = 5
    lw_high_res = 3
    label1 = "22 isotopes"
    label2 = "128 isotopes"


    fig = plt.figure(figsize=(30, 15))
    gs = gridspec.GridSpec(120, 120)

    ax = fig.add_subplot(gs[:40, :])
    bx = fig.add_subplot(gs[40:80, :])
    cx = fig.add_subplot(gs[80:, :])

    plot_rho(pfile1_low_res, ax, c=c1, lw=lw_low_res, ls=ls_low_res)
    plot_rho(pfile1_high_res, ax, c=c1, lw=lw_high_res, ls=ls_high_res, label=label1)

    if pfile2_low_res is not None:
        plot_rho(pfile2_low_res, ax, c=c2, lw=lw_low_res, ls=ls_low_res)
        plot_rho(pfile2_high_res, ax, c=c2, lw=lw_high_res, ls=ls_high_res, label=label2)

    plot_ye(pfile1_low_res, bx, c=c1, lw=lw_low_res, ls=ls_low_res)
    plot_ye(pfile1_high_res, bx, c=c1, lw=lw_high_res, ls=ls_high_res, label=label1)

    if pfile2_low_res is not None:
        plot_ye(pfile2_low_res, bx, c=c2, lw=lw_low_res, ls=ls_low_res)
        plot_ye(pfile2_high_res, bx, c=c2, lw=lw_high_res, ls=ls_high_res, label=label2)

    plot_j(pfile1_low_res, cx, c=c1, lw=lw_low_res, ls=ls_low_res)
    plot_j(pfile1_high_res, cx, c=c1, lw=lw_high_res, ls=ls_high_res, label=label1)

    if pfile2_low_res is not None:
        plot_j(pfile2_low_res, cx, c=c2, lw=lw_low_res, ls=ls_low_res)
        plot_j(pfile2_high_res, cx, c=c2, lw=lw_high_res, ls=ls_high_res, label=label2)

    ax.set_ylabel(r"$\log_{10}(\rho/\mathrm{[g \ cm^{-3}]})$")
    bx.set_ylabel(r"$Y_e$")
    cx.set_ylabel(r"$\log_{10}(j\ \mathrm{cm^2\ s^{-1}})$")
    # cx.set_xlabel(r"$m \ [M_\odot]$")
    cx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")

    ax.set_xlim(cx.get_xlim())
    bx.set_xlim(cx.get_xlim())
    bx.plot(np.nan, np.nan, ls=ls_high_res, lw=lw_high_res, c='k', label="high res")
    bx.plot(np.nan, np.nan, ls=ls_low_res, lw=lw_low_res, c='k', label="nominal res")
    bx.legend(ncol=2)
    plt.show()


if __name__ == "__main__":
    root = "../MESA_results/"

    small_net1 = root + '40_rot0.6_small_net/LOGS1/'
    pfile_small_net1 = small_net1+'CHE_single_core_collapse.data'
    small_net2 = root + 'res_test_small/mdc0.75_mtc0.75_mdchighT3/LOGS1/'
    pfile_small_net2 = small_net2+'CHE_single_core_collapse.data'
    small_net3 = root + 'res_test_small/mdc0.75_mtc0.75_mdchighT1.5/LOGS1/'
    pfile_small_net3 = small_net3+'CHE_single_core_collapse.data'
    small_net4 = root + 'res_test_small/mdc0.75_mtc0.75_mdchighT0.75/LOGS1/'
    pfile_small_net4 = small_net4+'CHE_single_core_collapse.data'
    small_net5 = root + 'res_test_small/mdc0.75_mtc0.75_mdchighT0.2/LOGS1/'
    pfile_small_net5 = small_net5+'CHE_single_core_collapse.data'

    fig = plt.figure(figsize=(30, 15))
    gs = gridspec.GridSpec(120, 120)

    ax = fig.add_subplot(gs[:40, :])
    bx = fig.add_subplot(gs[40:80, :])
    cx = fig.add_subplot(gs[80:, :])

    c1 = 'C0'
    c2 = 'C1'
    c3 = 'C2'
    c4 = 'C3'
    c5 = 'pink'


    legend1 = "1, 1, 3"
    legend2 = "0.75, 0.75, 3"
    legend3 = "0.75, 0.75, 1.5"
    legend4 = "0.75, 0.75, 0.75"
    legend5 = "0.75, 0.75, 0.2"

    plot_rho(pfile_small_net1, ax, c=c1, lw=5, label=legend1)
    plot_rho(pfile_small_net2, ax, c=c2, lw=4, label=legend2)
    plot_rho(pfile_small_net3, ax, c=c3, label=legend3)
    plot_rho(pfile_small_net4, ax, c=c4, lw=2, label=legend4)
    plot_rho(pfile_small_net5, ax, c=c5, lw=1, label=legend5)
    ax.legend(ncol=4, title="mesh, time, mesh highT", fontsize=20)


    plot_ye(pfile_small_net1, bx, c=c1, lw=5)
    plot_ye(pfile_small_net2, bx, c=c2, lw=4)
    plot_ye(pfile_small_net3, bx, c=c3)
    plot_ye(pfile_small_net4, bx, c=c4, lw=2)
    plot_ye(pfile_small_net5, bx, c=c5, lw=1)

    plot_j(pfile_small_net1, cx, c=c1, lw=5)
    plot_j(pfile_small_net2, cx, c=c2, lw=4)
    plot_j(pfile_small_net3, cx, c=c3)
    plot_j(pfile_small_net4, cx, c=c4, lw=2)
    plot_j(pfile_small_net5, cx, c=c5, lw=1)


    ax.set_ylabel(r"$\log_{10}(\rho/\mathrm{[g \ cm^{-3}]})$")
    bx.set_ylabel(r"$Y_e$")
    cx.set_ylabel(r"$\log_{10}(j\ \mathrm{cm^2\ s^{-1}})$")
    cx.set_xlabel(r"$m \ [M_\odot]$")

    fig.align_ylabels()
    plt.show()

    print(legend1)
    print(f"{get_Fe_core_mass(pfile_small_net1):.2f}")
    print(legend2)
    print(f"{get_Fe_core_mass(pfile_small_net2):.2f}")
    print(legend3)
    print(f"{get_Fe_core_mass(pfile_small_net3):.2f}")
    print(legend4)
    print(f"{get_Fe_core_mass(pfile_small_net4):.2f}")
    print(legend5)
    print(f"{get_Fe_core_mass(pfile_small_net5):.2f}")
