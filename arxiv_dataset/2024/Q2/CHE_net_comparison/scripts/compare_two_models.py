import numpy as np
from MESAreader import getSrcCol, clight, G_cgs, Msun, Rsun_cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from preCC_profile import calculate_lso_radius_and_spin_parameter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_percent_error_Ye(pfile1, pfile2):
    """Use q=m/Mtot as independent coordinate for interpolation."""
    src1, col1 = getSrcCol(pfile1)
    ye1 = src1[::-1, col1.index("ye")]
    logR1 = np.log10((10.0 ** src1[::-1, col1.index("logR")]) * Rsun_cm)
    # flip this to have monotonic increase
    q1 = src1[::-1, col1.index("mass")] / src1[0, col1.index("mass")]

    src2, col2 = getSrcCol(pfile2)
    ye2 = src2[::-1, col2.index("ye")]
    logR2 = np.log10((10.0 ** src2[::-1, col2.index("logR")]) * Rsun_cm)
    q2 = src2[::-1, col2.index("mass")] / src2[0, col2.index("mass")]

    x = np.linspace(0, 1, 2 * max(len(q1), len(q2)))

    ye1_interp = np.interp(x, q1, ye1)
    ye2_interp = np.interp(x, q2, ye2)
    log_r1_interp = np.interp(x, q1, logR1)
    log_r2_interp = np.interp(x, q2, logR2)

    delta = (ye1_interp - ye2_interp) / ye2_interp
    # flip back
    delta = delta[::-1]
    ye1_interp = ye1_interp[::-1]
    ye2_interp = ye2_interp[::-1]
    log_r1_interp = log_r1_interp[::-1]
    log_r2_interp = log_r2_interp[::-1]
    return (delta, ye1_interp, ye2_interp, log_r1_interp, log_r2_interp)


def plot_j(pfile, ax, shade=False, **kwargs):
    src, col = getSrcCol(pfile)
    j_specific = src[:, col.index("j_rot")]  # r*r*omega
    logR = np.log10((10.0 ** src[:, col.index("logR")]) * Rsun_cm)
    m = src[:, col.index("mass")]
    a = clight * j_specific / (G_cgs * m * Msun)
    r_ISCO, j_ISCO = calculate_lso_radius_and_spin_parameter(m, a)
    if shade:
        r_ISCO_nonrot, j_ISCO_nonrot = calculate_lso_radius_and_spin_parameter(m, np.zeros(len(m)))
        r_ISCO_maxrot, j_ISCO_maxrot = calculate_lso_radius_and_spin_parameter(m, np.ones(len(m)))
        ax.fill_between(logR, np.log10(j_ISCO_nonrot), np.log10(j_ISCO_maxrot), color=kwargs["c"], alpha=0.25, zorder=0)
    ax.plot(logR, np.log10(j_specific), **kwargs)
    ax.plot(logR, np.log10(j_ISCO), c=kwargs['c'], ls='--')
    disk = j_specific >= j_ISCO
    ax.scatter(logR[disk], np.log10(j_specific[disk]), marker="o", s=50, **kwargs)


def plot_ye(pfile, ax, **kwargs):
    src, col = getSrcCol(pfile)
    ye = src[:, col.index("ye")]
    logR = np.log10((10.0 ** src[:, col.index("logR")]) * Rsun_cm)
    ax.plot(logR, ye, **kwargs)


def make_one_HRD(hfile, ax, show_interp=True, **kwargs):
    src, col = getSrcCol(hfile)
    logT = src[:, col.index("log_Teff")]
    logL = src[:, col.index("log_L")]
    ax.plot(logT, logL, **kwargs)
    if show_interp:
        t = src[:, col.index("star_age")]
        t_interp = np.arange(0, max(t), 1e5)
        logL_interp = np.interp(t_interp, t, logL)
        logT_interp = np.interp(t_interp, t, logT)
        ax.scatter(logT_interp, logL_interp, marker='o', s=50, c=kwargs['c'], zorder=kwargs['zorder'])


def plot_rho_vs_logR(pfile, ax, **kwargs):
    src, col = getSrcCol(pfile)
    logrho = src[:, col.index("logRho")]
    logR = np.log10((10.0 ** src[:, col.index("logR")]) * Rsun_cm)
    ax.plot(logR, logrho, **kwargs)


def custom_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)
    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)
    return pp, p1, p2


def compare_two_models(LOGS1, LOGS2, label1="22 isotopes", label2="128 isotopes", fig_name=None):
    fig = plt.figure(figsize=(30, 15))
    gs = gridspec.GridSpec(120, 120)
    HRD_ax = fig.add_subplot(gs[:, 80:])
    ax = fig.add_subplot(gs[45:75, :68])
    right_ax = ax.twinx()
    bx = fig.add_subplot(gs[75:, :68])
    cx = fig.add_subplot(gs[:45, :68])

    hfile1 = LOGS1 + 'history.data'
    hfile2 = LOGS2 + 'history.data'

    pfile1 = LOGS1 + "CHE_single_core_collapse.data"
    pfile2 = LOGS2 + "CHE_single_core_collapse.data"

    c1 = "b"
    c2 = "orange"

    make_one_HRD(hfile1, ax=HRD_ax, c=c1, label=label1, zorder=10)
    make_one_HRD(hfile2, ax=HRD_ax, c=c2, label=label2, zorder=11)
    HRD_ax.invert_xaxis()
    HRD_ax.legend(loc="center left", handletextpad=0.5)
    HRD_ax.set_xlabel(r"$\log_{10}(T_\mathrm{eff}/[K])$")
    HRD_ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")

    bx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
    cx.set_ylabel(r"$\log_{10}(\rho/\mathrm{[g\ cm^{-3}]})$")

    # percentage error
    delta, ye1_interp, ye2_interp, log_r1_interp, log_r2_interp = get_percent_error_Ye(pfile1, pfile2)
    # sanity check on interpolation
    # ax.plot(log_r1_interp, ye1_interp, c=c1, marker='x')
    # ax.plot(log_r2_interp, ye2_interp, c=c2, marker='x')
    right_ax.plot(log_r1_interp, delta * 100, c='r', ls='-')
    right_ax.set_ylabel(r"$\Delta [\%]$", color='r')
    right_ax.spines['right'].set_color('r')
    right_ax.tick_params(colors='r')
    right_ax.tick_params(colors='r', which='minor')
    plot_ye(pfile1, ax, c=c1, label=label1, zorder=2)
    plot_ye(pfile2, ax, c=c2, label=label2, zorder=2)
    ax.set_xticklabels([])
    ax.set_ylabel(r"$Y_e=\sum X_i Z_i/A_i$")

    plot_j(pfile1, bx, c=c1, zorder=1)
    plot_j(pfile2, bx, c=c2, zorder=2)
    bx.plot(np.nan, np.nan, c='k', ls='-', label=r"$j$")
    bx.plot(np.nan, np.nan, c='k', ls='--', label=r"$j_\mathrm{ISCO}(m, j)$")
    bx.scatter(np.nan, np.nan, c='k', ls='-', marker='o', label=r"disk")
    bx.legend(loc="lower right", handleheight=0.5)
    bx.set_ylabel(r"$\log_{10}(j / [\mathrm{cm^2\ s^{-1}}])$")

    ax.set_xlim(bx.get_xlim())
    cx.set_xlim(bx.get_xlim())
    cxins = inset_axes(
        cx,
        width="50%",
        height="75%",
        bbox_to_anchor=(5.95, -7.8, 5, 15),
        bbox_transform=cx.transData,
        loc='lower left',
    )
    custom_mark_inset(cx, cxins, loc1a=4, loc1b=4, lw=2, zorder=10)
    plot_rho_vs_logR(pfile1, cxins, c=c1, zorder=1)
    plot_rho_vs_logR(pfile2, cxins, c=c2, zorder=2)
    cxins.set_xlim(8.0, 9)
    cxins.set_ylim(5, 8.5)
    plot_rho_vs_logR(pfile1, cx, c=c1, zorder=1)
    plot_rho_vs_logR(pfile2, cx, c=c2, zorder=2)
    cx.set_xticklabels([])

    fig.align_ylabels()
    if fig_name is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(fig_name)
        print("saved fig at", fig_name)


if __name__ == "__main__":
    root = "../MESA_results/"
    small = root + '40_rot0.6_small_net/'
    LOGS_small = small + "LOGS1/"
    large = root + '40_rot0.6_large_net/'
    LOGS_large = large + 'LOGS1/'
    fig_name = root + "../manuscript/figures/comparison.pdf"
    compare_two_models(LOGS_small, LOGS_large, fig_name=fig_name)
