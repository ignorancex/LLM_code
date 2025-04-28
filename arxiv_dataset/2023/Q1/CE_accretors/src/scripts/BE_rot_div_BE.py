from MESAreader import getSrcCol
from lib_plot_bin import plot_ratio_BErot_BE_m, get_ax_from_pfile, Rsun_cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import warnings
import paths


def grid_erot_fraction_BE(fig_name=None):
    """
    Parameters:
    ----------
    fig_name: `string` location where to save the figure
    """
    root = paths.data / "MESA_output/"
    # root_eng = root / "engineered_stars/same_core/"
    root_accretors = root / "binaries/Z_0.0019/"

    s1 = str(root) + "/single_stars/Z_0.0019/18_rot0.0/LOGS/"
    # engineered_grid1 = sorted(
    #     glob.glob(str(root_eng) + "/grid18/*.*/LOGS/"), key=sorter_engineered_profiles
    # )
    b1 = (
        str(root_accretors)
        + "/m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )

    s2 = str(root) + "/single_stars/Z_0.0019/20_rot0.0/LOGS/"
    # engineered_grid2 = sorted(
    #     glob.glob(str(root_eng) + "/grid20/*.*/LOGS/"), key=sorter_engineered_profiles
    # )
    b2 = (
        str(root_accretors)
        + "/m1_20.0000_m2_17.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )

    s3 = str(root) + "/single_stars/Z_0.0019/36_rot0.0/LOGS/"
    # engineered_grid3 = sorted(
    #     glob.glob(str(root_eng) + "/grid36/*.*/LOGS/"), key=sorter_engineered_profiles
    # )
    b3 = (
        str(root_accretors)
        + "/m1_38.0000_m2_30.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )
    # get default colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(120, 150)
    # accretor15, compare to 18Msun
    ax1 = fig.add_subplot(gs[:20, :50])
    ax2 = fig.add_subplot(gs[20:40, :50])
    ax3 = fig.add_subplot(gs[40:60, :50])
    ax4 = fig.add_subplot(gs[60:80, :50])
    ax5 = fig.add_subplot(gs[80:100, :50])

    axes = [ax1, ax2, ax3, ax4, ax5]
    for ax in axes:
        ax.set_xlim(8.2, 14)
        # ax.set_ylim(-0.05, 2.5)
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
    ax3.set_ylabel(r"$BE(\alpha_\mathrm{rot}=1.0)/BE(\alpha_\mathrm{rot}=0.0)$")

    profiles = sorted(glob.glob(b1 + "/*Rsun.data"))
    for pfile in profiles:
        string = pfile.split("/")[-1]
        ax = get_ax_from_pfile(pfile, axes)
        ratio = plot_ratio_BErot_BE_m(pfile, ax, c=colors[0])

    # # 20 Msun
    bx1 = fig.add_subplot(gs[:20, 50:100])
    bx2 = fig.add_subplot(gs[20:40, 50:100])
    bx3 = fig.add_subplot(gs[40:60, 50:100])
    bx4 = fig.add_subplot(gs[60:80, 50:100])
    bx5 = fig.add_subplot(gs[80:100, 50:100])

    bxes = [bx1, bx2, bx3, bx4, bx5]
    for bx in bxes:
        bx.set_xlim(8.2, 14)
        # bx.set_ylim(-0.05, 2.5)
        bx.set_yticklabels([])
        bx.set_yticklabels([])
        if bx == bxes[-1]:
            bx.axhline(1, 0, 1, lw=2, ls=":", c="#808080")
            bx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
        else:
            bx.set_xticklabels([])

    profiles = sorted(glob.glob(b2 + "/*Rsun.data"))  # , key=get_age_from_profile)
    for pfile in profiles:
        string = pfile.split("/")[-1]
        bx = get_ax_from_pfile(pfile, bxes)
        ratio = plot_ratio_BErot_BE_m(pfile, bx, c=colors[1])
        # print(pfile.split("/")[-1], pfile.split("/")[-3], max(ratio), min(ratio))

    # 30
    cx1 = fig.add_subplot(gs[:20, 100:])
    cx2 = fig.add_subplot(gs[20:40, 100:])
    cx3 = fig.add_subplot(gs[40:60, 100:])
    cx4 = fig.add_subplot(gs[60:80, 100:])
    cx5 = fig.add_subplot(gs[80:100, 100:])

    cxes = [cx1, cx2, cx3, cx4, cx5]
    for cx in cxes:
        cx.set_xlim(8.2, 14)
        # cx.set_ylim(-0.05, 2.5)
        # cx.axhline(1, 0, 1, lw=2, ls=":", c="#808080")
        cx.set_yticklabels([])
        if cx != cxes[-1]:
            cx.set_xticklabels([])
        else:
            cx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")

    profiles = sorted(glob.glob(b3 + "/*Rsun.data"))  # , key=get_age_from_profile)
    for pfile in profiles:
        string = pfile.split("/")[-1]
        cx = get_ax_from_pfile(pfile, cxes)
        ratio = plot_ratio_BErot_BE_m(pfile, cx, c=colors[2])
        # print(pfile.split("/")[-1], pfile.split("/")[-3], max(ratio), min(ratio))

    for cx in cxes:
        dx = cx.twinx()
        if cx == cxes[0]:
            label = "$100\,R_\odot$"
            X = 100 * Rsun_cm
        if cx == cxes[1]:
            label = "$200\,R_\odot$"
            X = 200 * Rsun_cm
        if cx == cxes[2]:
            label = "$300\,R_\odot$"
            X = 300 * Rsun_cm
        if cx == cxes[3]:
            label = "$500\,R_\odot$"
            X = 500 * Rsun_cm
        if cx == cxes[4]:
            label = "$1000\,R_\odot$"
            X = 1000 * Rsun_cm
        dx.set_ylabel(label)
        cx.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=0)
        dx.set_ylim(cx.get_ylim())
        dx.set_yticks(cx.get_yticks())
        dx.set_yticklabels([])
        ax = axes[cxes.index(cx)]
        if ax != axes[-1]:  # skip last panel
            ax.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=0)
        bx = bxes[cxes.index(cx)]
        if bx != bxes[-1]:  # skip last panel
            bx.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=0)

    ax1.set_title(r"$M_2=15\rightarrow 18\,M_\odot$", size=30)
    bx1.set_title(r"$M_2=17\rightarrow 20\,M_\odot$", size=30)
    cx1.set_title(r"$M_2=30\rightarrow 36\,M_\odot$", size=30)

    # axes[-1].plot(np.nan, np.nan, c="r", ls="-", label="ratio to single")
    # axes[-1].legend(handletextpad=0.1, handlelength=0.75, loc="center")

    if fig_name:
        plt.savefig(fig_name)
    else:
        plt.show()


if __name__ == "__main__":
    # three_masses_grid(fig_name='grid_ratios.pdf')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_erot_fraction_BE()
