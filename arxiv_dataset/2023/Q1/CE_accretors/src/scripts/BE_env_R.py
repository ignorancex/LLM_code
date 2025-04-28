from MESAreader import getSrcCol
from lib_plot_bin import plot_ratio_BE_r, get_ax_from_pfile, interpolate_BE_env, Rsun_cm
from lib_engineered import get_M_boundary
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import warnings
import paths


if __name__ == "__main__":
    root = paths.data / "MESA_output/"

    root_accretors = root / "binaries/Z_0.0019/"
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

    nonrot36 = root / "single_stars/Z_0.0019/36_rot0.0/LOGS/"
    nonrot20 = root / "single_stars/Z_0.0019/20_rot0.0/LOGS/"
    nonrot18 = root / "single_stars/Z_0.0019/18_rot0.0/LOGS/"
    single_stars = [str(nonrot18), str(nonrot20), str(nonrot36)]

    # get default colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig = plt.figure(figsize=(8, 14))
    gs = gridspec.GridSpec(124, 100)
    ax1 = fig.add_subplot(gs[:25, :])
    bx1 = fig.add_subplot(gs[25:40, :])
    ax2 = fig.add_subplot(gs[42:67, :])
    bx2 = fig.add_subplot(gs[67:82, :])
    ax3 = fig.add_subplot(gs[84:109, :])
    bx3 = fig.add_subplot(gs[109:, :])

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-1, 1200)
        ax.set_yscale("log")
        ax.set_xticklabels([])
        ax.set_ylabel(r"$BE_\mathrm{env} \ [\mathrm{erg}]$")

    for bx in [bx1, bx2, bx3]:
        bx.set_xlim(-1, 1200)
        bx.axhline(1, 0, 1, c="#808080", ls="--", lw=2, zorder=0)
        bx.set_ylim(0.2, 1.8)
        bx.set_yticks([0.5, 1, 1.5])
        bx.set_yticks([0.25, 0.5, 0.75, 1.25, 1.5])
        if bx != bx3:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel(r"$R \ [R_\odot]$")

    for f in single_stars:
        c = colors[single_stars.index(f)]
        if "18" in f:
            b = b1
            ax = ax3
            bx = bx3
            label_single = "single $18\,M_\odot$"
            label = "accretor"
        elif "20" in f:
            b = b2
            ax = ax2
            bx = bx2
            label_single = "single $20\,M_\odot$"
            label = "accretor"
        elif "36" in f:
            b = b3
            ax = ax1
            bx = bx1
            label_single = "single $36\,M_\odot$"
            label = "accretor"
        # single star
        src, col = getSrcCol(f + "/history.data")
        t = src[:, col.index("star_age")]
        r = src[:, col.index("radius")]
        env_BE = (
            -1.0 * src[:, col.index("envelope_binding_energy")]
        )  # MESA assumes alpha_th=1 and flips the sign
        logTc = src[:, col.index("log_center_T")]
        # highlight increasing radii
        i_increasing_single = [i == np.argmax(r[:i+1]) for i in range(len(r))]
        # separate chunks increasing and non-increasing w.r.t. past maximum
        x_increasing = [r[i] if i_increasing_single[i] else np.nan for i in range(len(i_increasing_single))]
        x_decreasing = [r[i] if not i_increasing_single[i] else np.nan for i in range(len(i_increasing_single))]
        ax.plot(x_decreasing, env_BE, ls=":", lw=1, c=c, zorder=0)
        ax.plot(x_increasing, env_BE, ls=":", lw=3, c=c, label=label_single, zorder=1)
        # accretor star
        src, col = getSrcCol(b + "history.data")
        x_center = src[:, col.index("center_h1")]
        y_center = src[:, col.index("center_he4")]
        # iHe_depl = np.argmin(np.absolute(y_center - 1e-4))
        # iTAMS = np.argmin(np.absolute(x_center - 1e-4))
        env_BE_accretor = (
            -1.0 * src[:, col.index("envelope_binding_energy")]
        )  # MESA assumes alpha_th=1
        t_accretor = src[:, col.index("star_age")]
        r_accretor = src[:, col.index("radius")]
        logTc_accretor = src[:, col.index("log_center_T")]
        # highlight increasing radii
        i_increasing_accretor = [i == np.argmax(r_accretor[:i+1]) for i in range(len(r_accretor))]
        # separate chunks increasing and non-increasing w.r.t. past maximum
        x_increasing_accretor = [r_accretor[i] if i_increasing_accretor[i] else np.nan for i in range(len(i_increasing_accretor))]
        x_decreasing_accretor = [r_accretor[i] if not i_increasing_accretor[i] else np.nan for i in range(len(i_increasing_accretor))]
        ax.plot(x_decreasing_accretor, env_BE_accretor, ls="-", lw=1, c=c, zorder=0)
        ax.plot(x_increasing_accretor, env_BE_accretor, ls="-", lw=3, c=c, label=label, zorder=1)
        # # now interpolate
        env_BE_accretor_interp = interpolate_BE_env(logTc, logTc_accretor, env_BE_accretor)
        # # calculate ratio
        ratio = env_BE_accretor_interp / env_BE
        bx.plot(x_increasing, ratio, lw=3, c=c)
        bx.plot(x_decreasing, ratio, lw=1, c=c)


    # deal with legends
    for ax in [ax1, ax2, ax3]:
        ax.legend(
            loc="best", fontsize=20, labelspacing=0.1, handletextpad=0.1)

    fig.align_labels()
    plt.savefig(paths.figures / "BE_env_R.pdf")
