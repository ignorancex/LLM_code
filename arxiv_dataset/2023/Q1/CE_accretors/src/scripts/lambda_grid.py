from MESAreader import getSrcCol
from lib_plot_bin import plot_lambda_mass, plot_lambda_at_one_radius, Rsun_cm
from lib_engineered import get_M_boundary, sorter_engineered_profiles
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import paths


def grid_lambdas(
    b1,
    b2,
    b3,
    nonrot1,
    nonrot2,
    nonrot3,
    grid_folders1,
    grid_folders2,
    grid_folders3,
    plot_func=plot_lambda_mass,
    legend=False,
    fig_name=None,
):
    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(120, 150)
    # # 15Msun accretor
    ax1 = fig.add_subplot(gs[:20, :50])
    ax2 = fig.add_subplot(gs[20:40, :50])
    ax3 = fig.add_subplot(gs[40:60, :50])
    ax4 = fig.add_subplot(gs[60:80, :50])
    ax5 = fig.add_subplot(gs[80:100, :50])
    axes = [ax1, ax2, ax3, ax4, ax5]
    for ax in axes:
        ax.set_xlim(0, 18.5)
        ax.set_yscale("log")
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$m\ [M_\odot]$")
    ax1.set_ylim(5e-3, 1.85)
    ax2.set_ylim(5e-3, 1.85)
    ax3.set_ylim(5e-3, 1.85)
    ax4.set_ylim(5e-3, 1.85)
    ax5.set_ylim(5e-3, 2.3)
    plot_lambda_at_one_radius(
        ax1,
        "100Rsun.data",
        grid_folders=grid_folders1,
        accretor=b1,
        nonrot=nonrot1,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        ax2,
        "200Rsun.data",
        grid_folders=grid_folders1,
        accretor=b1,
        nonrot=nonrot1,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        ax3,
        "300Rsun.data",
        grid_folders=grid_folders1,
        accretor=b1,
        nonrot=nonrot1,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        ax4,
        "500Rsun.data",
        grid_folders=grid_folders1,
        accretor=b1,
        nonrot=nonrot1,
        plot_func=plot_func,
    )
    # # 20 Msun
    bx1 = fig.add_subplot(gs[:20, 50:100])
    bx2 = fig.add_subplot(gs[20:40, 50:100])
    bx3 = fig.add_subplot(gs[40:60, 50:100])
    bx4 = fig.add_subplot(gs[60:80, 50:100])
    bx5 = fig.add_subplot(gs[80:100, 50:100])
    bxes = [bx1, bx2, bx3, bx4, bx5]
    for bx in bxes:
        bx.set_xlim(0, 21.7)
        bx.set_yscale("log")
        bx.set_ylim(axes[bxes.index(bx)].get_ylim())
        bx.set_yticklabels([])
        if bx != bxes[-1]:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel(r"$m\ [M_\odot]$")
    plot_lambda_at_one_radius(
        bx1,
        "100Rsun.data",
        grid_folders=grid_folders2,
        accretor=b2,
        nonrot=nonrot2,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        bx2,
        "200Rsun.data",
        grid_folders=grid_folders2,
        accretor=b2,
        nonrot=nonrot2,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        bx3,
        "300Rsun.data",
        grid_folders=grid_folders2,
        accretor=b2,
        nonrot=nonrot2,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        bx4,
        "500Rsun.data",
        grid_folders=grid_folders2,
        accretor=b2,
        nonrot=nonrot2,
        plot_func=plot_func,
    )
    # 30
    cx1 = fig.add_subplot(gs[:20, 100:])
    cx2 = fig.add_subplot(gs[20:40, 100:])
    cx3 = fig.add_subplot(gs[40:60, 100:])
    cx4 = fig.add_subplot(gs[60:80, 100:])
    cx5 = fig.add_subplot(gs[80:100, 100:])
    cxes = [cx1, cx2, cx3, cx4, cx5]
    for cx in cxes:
        cx.set_xlim(0, 36.3)
        cx.set_yscale("log")
        cx.set_ylim(axes[cxes.index(cx)].get_ylim())
        cx.set_yticklabels([])
        if cx != cxes[-1]:
            cx.set_xticklabels([])
        else:
            cx.set_xlabel(r"$m\ [M_\odot]$")
    plot_lambda_at_one_radius(
        cx1,
        "100Rsun.data",
        grid_folders=grid_folders3,
        accretor=b3,
        nonrot=nonrot3,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        cx2,
        "200Rsun.data",
        grid_folders=grid_folders3,
        accretor=b3,
        nonrot=nonrot3,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        cx3,
        "300Rsun.data",
        grid_folders=grid_folders3,
        accretor=b3,
        nonrot=nonrot3,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        cx4,
        "500Rsun.data",
        grid_folders=grid_folders3,
        accretor=b3,
        nonrot=nonrot3,
        plot_func=plot_func,
    )
    plot_lambda_at_one_radius(
        cx5,
        "1000Rsun.data",
        grid_folders=grid_folders3,
        accretor=b3,
        nonrot=nonrot3,
        plot_func=plot_func,
    )
    ax3.set_ylabel(r"$\lambda_\mathrm{CE} = (GM(M-m)/R)/BE(m, \alpha_\mathrm{th}=1.0)$")
    for cx in cxes:
        dx = cx.twinx()
        dx.set_yscale("log")
        dx.set_ylim(cx.get_ylim())
        # dx.set_yticks(cx.get_yticks())
        dx.set_yticklabels([])
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
    # titles and legends
    ax1.set_title(r"$M_2=15\rightarrow 18\,M_\odot$", size=30)
    bx1.set_title(r"$M_2=17\rightarrow 20\,M_\odot$", size=30)
    cx1.set_title(r"$M_2=30\rightarrow 36\,M_\odot$", size=30)
    if legend:
        bx5.plot(np.nan, np.nan, c="orange", lw=3, ls="-", label="accretor")
        bx5.plot(np.nan, np.nan, c="r", lw=3, ls="-", label="single star")
        bx5.legend()

    if fig_name:
        plt.savefig(fig_name)


if __name__ == "__main__":
    # locations, to be fixed when uploading on zenodo
    root = paths.data / "MESA_output/"
    root_accretors = root / "binaries/Z_0.0019/"
    root_simplified = str(root / "engineered_stars/same_core/")

    b1 = (
        str(root_accretors)
        + "/m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )
    b2 = (
        str(root_accretors)
        + "/m1_20.0000_m2_17.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )
    b3 = (
        str(root_accretors)
        + "/m1_38.0000_m2_30.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/LOGS2/"
    )

    nonrot36 = str(root / "single_stars/Z_0.0019/36_rot0.0/LOGS") + "/"
    nonrot20 = str(root / "single_stars/Z_0.0019/20_rot0.0/LOGS") + "/"
    nonrot18 = str(root / "single_stars/Z_0.0019/18_rot0.0/LOGS") + "/"

    root_grid18 = root_simplified + "/grid18/"
    root_grid20 = root_simplified + "/grid20/"
    root_grid36 = root_simplified + "/grid36/"
    grid_folders18 = sorted(
        glob.glob(root_grid18 + "/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    grid_folders20 = sorted(
        glob.glob(root_grid20 + "/*.*/LOGS/"), key=sorter_engineered_profiles
    )
    grid_folders36 = sorted(
        glob.glob(root_grid36 + "/*.*/LOGS/"), key=sorter_engineered_profiles
    )

    fig_name = paths.figures / "lambda_grid.pdf"

    grid_lambdas(
        b1,
        b2,
        b3,
        nonrot18,
        nonrot20,
        nonrot36,
        grid_folders18,
        grid_folders20,
        grid_folders36,
        plot_func=plot_lambda_mass,
        legend=True,
        fig_name=fig_name,
    )
