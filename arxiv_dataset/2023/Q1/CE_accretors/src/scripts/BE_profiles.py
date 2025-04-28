from lib_plot_bin import plot_BE_r, get_ax_from_pfile, Rsun_cm
from lib_engineered import sorter_engineered_profiles
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import paths
import warnings


def grid_BE_profiles(fig_name=None):
    """
    Parameters:
    ----------
    fig_name: `string` location where to save the figure
    """

    root = paths.data / "MESA_output/"
    root_eng = root / "engineered_stars/same_core/"
    root_accretors = root / "binaries/Z_0.0019/"

    s1 = str(root) + "/single_stars/Z_0.0019/18_rot0.0/LOGS/"
    single_grid1 = sorted(
        glob.glob(str(root_eng) + "/grid18/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    b1 = (
        str(root_accretors)
        + "/m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )

    s2 = str(root) + "/single_stars/Z_0.0019/20_rot0.0/LOGS/"
    single_grid2 = sorted(
        glob.glob(str(root_eng) + "/grid20/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    b2 = (
        str(root_accretors)
        + "/m1_20.0000_m2_17.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )

    s3 = str(root) + "/single_stars/Z_0.0019/36_rot0.0/LOGS/"
    single_grid3 = sorted(
        glob.glob(str(root_eng) + "/grid36/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    b3 = (
        str(root_accretors)
        + "/m1_38.0000_m2_30.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )

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
        ax.set_ylim(1e45, 1e51)
        ax.set_yscale("log")
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            # ax.set_yticklabels([])
            ax.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
    ax3.set_ylabel(r"$BE(m,\alpha_\mathrm{th}=1) \ [\mathrm{erg}]$")
    profiles = sorted(glob.glob(b1 + "/*Rsun.data"))  # , key=get_age_from_profile)
    for pfile_accretor in profiles:
        string = pfile_accretor.split("/")[-1]
        ax = get_ax_from_pfile(pfile_accretor, axes)
        plot_BE_r(
            pfile_accretor,
            ax,
            mark_he_core=True,
            alpha_th=1.0,
            c="orange",
            ls="-",
            zorder=10,
        )
        colors = plt.cm.viridis(np.linspace(0, 1, len(single_grid1)))
        for f in single_grid1:
            pfile_single = glob.glob(f + string)[0]  # +string)
            plot_BE_r(
                pfile_single,
                ax,
                alpha_th=1.0,
                c=colors[single_grid1.index(f)],
                ls="-",
                lw=2,
            )
        # now plot normal single star
        pfile_single = glob.glob(s1 + string)[0]
        plot_BE_r(
            pfile_single, ax, mark_he_core=True, alpha_th=1.0, c="r", ls="-", zorder=9
        )

    # # 20 Msun
    bx1 = fig.add_subplot(gs[:20, 50:100])
    bx2 = fig.add_subplot(gs[20:40, 50:100])
    bx3 = fig.add_subplot(gs[40:60, 50:100])
    bx4 = fig.add_subplot(gs[60:80, 50:100])
    bx5 = fig.add_subplot(gs[80:100, 50:100])

    bxes = [bx1, bx2, bx3, bx4, bx5]
    for bx in bxes:
        bx.set_xlim(8.2, 14)
        bx.set_ylim(1e45, 1e51)
        bx.set_yscale("log")
        bx.set_yticklabels([])
        if bx != bxes[-1]:
            bx.set_xticklabels([])
        else:
            bx.set_yticklabels([])
            bx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")

    profiles = sorted(glob.glob(b2 + "/*Rsun.data"))  # , key=get_age_from_profile)
    # for pfile_accretor in tqdm(profiles, desc="20Msun"):
    for pfile_accretor in profiles:
        # if ("10Rsun" in pfile_accretor) or ("20Rsun" in pfile_accretor):
        #     continue
        string = pfile_accretor.split("/")[-1]
        ax = get_ax_from_pfile(pfile_accretor, bxes)
        plot_BE_r(
            pfile_accretor,
            ax,
            mark_he_core=True,
            alpha_th=1.0,
            c="orange",
            ls="-",
            zorder=10,
        )
        colors = plt.cm.viridis(np.linspace(0, 1, len(single_grid2)))
        for f in single_grid2:
            pfile_single = glob.glob(f + string)[0]  # +string)
            plot_BE_r(
                pfile_single,
                ax,
                alpha_th=1.0,
                c=colors[single_grid2.index(f)],
                ls="-",
                lw=2,
            )
        # now plot normal single star
        pfile_single = glob.glob(s2 + string)[0]
        plot_BE_r(
            pfile_single, ax, mark_he_core=True, alpha_th=1.0, c="r", ls="-", zorder=9
        )

    # 30
    cx1 = fig.add_subplot(gs[:20, 100:])
    cx2 = fig.add_subplot(gs[20:40, 100:])
    cx3 = fig.add_subplot(gs[40:60, 100:])
    cx4 = fig.add_subplot(gs[60:80, 100:])
    cx5 = fig.add_subplot(gs[80:100, 100:])

    cxes = [cx1, cx2, cx3, cx4, cx5]
    for cx in cxes:
        cx.set_xlim(8.2, 14)
        cx.set_ylim(1e45, 1e51)
        cx.set_yscale("log")
        cx.set_yticklabels([])
        if cx != cxes[-1]:
            cx.set_xticklabels([])
        else:
            cx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")

    profiles = sorted(glob.glob(b3 + "/*Rsun.data"))  # , key=get_age_from_profile)
    for pfile_accretor in profiles:
        string = pfile_accretor.split("/")[-1]
        ax = get_ax_from_pfile(pfile_accretor, cxes)
        plot_BE_r(
            pfile_accretor,
            ax,
            mark_he_core=True,
            alpha_th=1.0,
            c="orange",
            ls="-",
            zorder=10,
        )
        colors = plt.cm.viridis(np.linspace(0, 1, len(single_grid3)))
        for f in single_grid3:
            pfile_single = glob.glob(f + string)[0]  # +string)
            plot_BE_r(
                pfile_single,
                ax,
                alpha_th=1.0,
                c=colors[single_grid3.index(f)],
                ls="-",
                lw=2,
            )
        # now plot normal single star
        pfile_single = glob.glob(s3 + string)[0]
        plot_BE_r(
            pfile_single, ax, mark_he_core=True, alpha_th=1.0, c="r", ls="-", zorder=9
        )

    # ticks
    for ax in axes + bxes + cxes:
        ax.set_yticks([1e46, 1e48, 1e50], minor=False)
        ax.set_yticks([1e45, 1e47, 1e49], minor=True)
        ax.set_yticklabels([], minor=True)

    for cx in cxes:
        dx = cx.twinx()
        if cx == cxes[0]:
            # bx.text(0.2, 0.1, , fontsize=30, transform=bx.transAxes, va="bottom", ha="right")
            label = "$100\,R_\odot$"
            X = 100 * Rsun_cm
        if cx == cxes[1]:
            # bx.text(0.2, 0.1, , fontsize=30, transform=bx.transAxes, va="bottom", ha="right")
            label = "$200\,R_\odot$"
            X = 200 * Rsun_cm
        if cx == cxes[2]:
            # bx.text(0.2, 0.1, , fontsize=30, transform=bx.transAxes, va="bottom", ha="right")
            label = "$300\,R_\odot$"
            X = 300 * Rsun_cm
        if cx == cxes[3]:
            # bx.text(0.2, 0.1, , fontsize=30, transform=bx.transAxes, va="bottom", ha="right")
            label = "$500\,R_\odot$"
            X = 500 * Rsun_cm
        if cx == cxes[4]:
            # bx.text(0.2, 0.1, , fontsize=30, transform=bx.transAxes, va="bottom", ha="right")
            label = "$1000\,R_\odot$"
            X = 1000 * Rsun_cm
        dx.set_ylabel(label)
        cx.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=0)
        ax = axes[cxes.index(cx)]
        if ax != axes[-1]:  # skip last panel
            ax.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=0)
        bx = bxes[cxes.index(cx)]
        if bx != bxes[-1]:  # skip last panel
            bx.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=0)
        dx.set_yscale(cx.get_yscale())
        dx.set_ylim(cx.get_ylim())
        dx.set_yticks([1e46, 1e48, 1e50], minor=False)
        dx.set_yticks([1e45, 1e47, 1e49], minor=True)
        dx.set_yticklabels([])
        dx.set_yticklabels([], minor=True)

    ax1.set_title(r"$M_2=15\rightarrow 18\,M_\odot$", size=30)
    bx1.set_title(r"$M_2=17\rightarrow 20\,M_\odot$", size=30)
    cx1.set_title(r"$M_2=30\rightarrow 36\,M_\odot$", size=30)

    # legends
    bxes[-1].plot(np.nan, np.nan, c="r", ls="-", label="single star")
    bxes[-1].plot(np.nan, np.nan, c="orange", ls="-", label="accretor")
    bxes[-1].legend(loc="center")

    if fig_name:
        plt.savefig(fig_name)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_BE_profiles(fig_name=paths.figures / "BE_profiles.pdf")
