import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Tuple
from matplotlib import lines, transforms
from matplotlib.ticker import ScalarFormatter
import matplotlib.patheffects as mpe

from zbands import z_bands_group, z_bands_cluster, line_annotate
from lookback import (
    redshift_from_lookback_time,
    inverse_hubble_time,
)

# Global constants
RUNS_LOCATION = "../data"
Z_TICKS = [1, 2, 3, 5, 8, 11, 16]
TIME_AXIS_TICKS = [0, 5, 8, 11, 13, 13.5]
PE = [
    mpe.Stroke(linewidth=2.5, foreground="w"),
    mpe.Stroke(foreground="w", alpha=0.6),
    mpe.Normal(),
]

# Load plot style
plt.style.use("mnras.mplstyle")

pe = [
    mpe.Stroke(linewidth=2.5, foreground="w"),
    mpe.Stroke(foreground="w", alpha=0.6),
    mpe.Normal(),
]

linestyles = [
    dict(lw=2, c="#488f31", path_effects=pe),
    dict(lw=2, c="k", path_effects=pe),
    dict(lw=2, c="#de425b", path_effects=pe),
]


def get_data(filename: str, run: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data and redshift for the given filename and run."""
    try:
        data = np.load(os.path.join(RUNS_LOCATION, run, f"{filename}.npy"))
        redshift = np.load(os.path.join(RUNS_LOCATION, run, "redshift.npy"))
        return redshift, data
    except FileNotFoundError:
        return np.zeros(1), np.zeros(1)


def make_time_evolution_canvas(ylabels: tuple[str]) -> tuple[plt.Figure, plt.Axes]:
    num_rows = len(ylabels)
    num_cols = 2
    zp1_low, zp1_high = 0.9, 11

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6.5, 6.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1] * num_rows},
        sharex=True,
    )
    fig.set_facecolor("white")

    axes[0, 0].set_title("Group", fontsize=11, pad=10)
    axes[0, 1].set_title("Cluster", fontsize=11, pad=10)

    # Draw redshift bands
    for z_idx, z_band in enumerate(z_bands_group):
        shaded_kwargs = dict(
            facecolor=z_band.color, edgecolor="none", alpha=0.4, zorder=0
        )
        i = 0
        ax = axes[0, i]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)

        ax.text(
            z_band.label_midpoint_log,
            1 - 0.1,
            z_band.name,
            ha="center",
            color="k",
            transform=trans,
            zorder=100,
            fontsize=7.5,
            va="center",
        )
        ax.text(
            z_band.label_midpoint_log,
            0.05,
            z_band.z_label,
            ha="center",
            va="bottom",
            color="k",
            transform=trans,
            zorder=100,
            rotation=90,
            fontsize=5,
        )

        ax = axes[1, i]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)
        ax.text(
            z_band.label_midpoint_log,
            1 - 0.05,
            z_band.time_label,
            rotation=90,
            fontsize=5,
            ha="center",
            va="top",
            color="k",
            transform=trans,
            zorder=100,
        )
        ax = axes[2, i]
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)

    for z_idx, z_band in enumerate(z_bands_cluster):
        shaded_kwargs = dict(
            facecolor=z_band.color, edgecolor="none", alpha=0.4, zorder=0
        )
        i = 1
        ax = axes[0, i]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)

        ax.text(
            z_band.label_midpoint_log,
            1 - 0.1,
            z_band.name,
            ha="center",
            color="k",
            transform=trans,
            zorder=100,
            fontsize=7.5,
            va="center",
        )
        ax.text(
            z_band.label_midpoint_log,
            0.05,
            z_band.z_label,
            ha="center",
            va="bottom",
            color="k",
            transform=trans,
            zorder=100,
            rotation=90,
            fontsize=5,
        )

        ax = axes[1, i]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)
        ax.text(
            z_band.label_midpoint_log,
            1 - 0.05,
            z_band.time_label,
            rotation=90,
            fontsize=5,
            ha="center",
            va="top",
            color="k",
            transform=trans,
            zorder=100,
        )
        ax = axes[2, i]
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)

    for i, ax in enumerate(axes.flat):
        ax.loglog()
        ax.grid(color="grey", which="major", alpha=0.3, lw=0.3, ls="--", zorder=0)

        # Add the redsift labels
        ax.set_xticks(Z_TICKS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticklabels([rf"${z - 1:.0f}$" for z in Z_TICKS])
        ax.set_xlim(zp1_low, zp1_high)

    # Add the time labels
    for ax in [axes[0, 0], axes[0, 1]]:
        ax1_time = ax.twiny()
        ax1_time.set_xscale("log")
        ax1_time.set_xlabel("Lookback time [Gyr]", labelpad=10)
        ax1_time.set_xlim(zp1_low, zp1_high)
        ax1_time.minorticks_off()
        ax1_time.set_xticks(
            [redshift_from_lookback_time(t) + 1 for t in TIME_AXIS_TICKS]
        )
        ax1_time.set_xticklabels([f"{t:.1f}" for t in TIME_AXIS_TICKS])

    for i, ylabel in enumerate(ylabels):
        axes[i, 0].set_ylabel(ylabel)

    quenched_limit = 1e-2
    bh_seed_mass = 1e4
    zp1_seed_label = 8.5
    for i in range(2):
        old_lims = axes[0, i].get_ylim()
        axes[0, i].set_ylim(old_lims[0], old_lims[1] * 3)

        axes[1, i].set_ylim(1e-3, None)
        # axes[1, i].axhspan(quenched_limit, quenched_limit * 0.8, xmax=0.7, fc='grey', ec='none',
        #                    alpha=0.9)
        # trans = transforms.blended_transform_factory(axes[1, i].transAxes, axes[1, i].transData)
        # axes[1, i].text(0.7, quenched_limit * 0.75, r'$\downarrow$ BCG quenched $\downarrow$',
        #                 transform=trans, ha='right', va='top', fontsize='medium', zorder=100,
        #                 color='k')
        axes[1, i].scatter(
            [1],
            [quenched_limit],
            c="grey",
            marker="v",
            alpha=1,
            ec="w",
            lw=0.7,
            zorder=200,
            s=80,
        )

        _ssfr_quench_limit_redshift = np.linspace(0.5, 7.5, 50)
        axes[1, i].plot(
            _ssfr_quench_limit_redshift + 1,
            [inverse_hubble_time(i) for i in _ssfr_quench_limit_redshift],
            c="grey",
            alpha=1,
            zorder=100,
            lw=2,
            ls="--",
            path_effects=pe,
        )
        _ssfr_quench_limit_redshift = np.linspace(0.5, 6.5, 50)
        axes[1, i].plot(
            _ssfr_quench_limit_redshift + 1,
            [inverse_hubble_time(i) / 3 for i in _ssfr_quench_limit_redshift],
            c="grey",
            alpha=1,
            zorder=100,
            lw=2,
            ls="-.",
            path_effects=pe,
        )

        (l,) = axes[1, i].plot(
            _ssfr_quench_limit_redshift + 1,
            [inverse_hubble_time(i) / 5 for i in _ssfr_quench_limit_redshift],
            c="grey",
            alpha=0,
            zorder=0,
            lw=1,
            ls="-.",
        )
        line_annotate(
            r"$\downarrow$ BCG quenched $\downarrow$",
            l,
            5,
            va="top",
            ha="center",
            zorder=100,
            xytext=(0, 0.05),
            textcoords="offset points",
        )

        axes[2, i].set_ylim(1e3, 1e10)
        top_pointer = 10 ** (
            (np.log10(axes[2, i].get_ylim()[0]) + np.log10(axes[2, i].get_ylim()[1]))
            * 0.5
        )
        top_pointer = 2e5
        axes[2, i].plot(
            [zp1_seed_label, zp1_seed_label],
            [bh_seed_mass, top_pointer],
            color="k",
            zorder=300,
            lw=0.85,
        )
        axes[2, i].scatter(
            [zp1_seed_label], [bh_seed_mass], fc="w", ec="k", lw=0.4, zorder=300, s=25
        )
        axes[2, i].text(
            zp1_seed_label,
            top_pointer * 1.7,
            f"BH seed",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize="medium",
        )

        axes[num_rows - 1, i].set_xlabel("Redshift")

    return fig, axes


if __name__ == "__main__":

    ylabels = (
        r"$M_{500}$ [M$_\odot$]",
        r"sSFR [Gyr$^{-1}$]",
        r"$M_{\rm cBH}$ [M$_\odot$]",
    )
    fig, axes = make_time_evolution_canvas(ylabels)
    ax_accretion_group = axes[2, 0].twinx()
    ax_accretion_group.set_yscale("log")
    ax_accretion_group.set_ylim(1e-3, 20)
    ax_accretion_group.minorticks_off()
    axes[2, 0].set_zorder(ax_accretion_group.get_zorder() + 1)
    axes[2, 0].patch.set_visible(False)

    ax_accretion_cluster = axes[2, 1].twinx()
    ax_accretion_cluster.set_yscale("log")
    ax_accretion_cluster.set_ylabel("BHMAR", labelpad=10)
    ax_accretion_cluster.set_ylim(1e-3, 60)
    ax_accretion_cluster.minorticks_off()
    ax_accretion_cluster.patch.set_facecolor("none")
    axes[2, 1].set_zorder(ax_accretion_cluster.get_zorder() + 1)
    axes[2, 1].patch.set_visible(False)

    linestyles_mdot = [
        dict(
            lw=1, ls=(0, (0.1, 2)), dash_capstyle="round", c="#488f31", path_effects=pe
        ),
        dict(lw=1, ls=(0, (0.1, 2)), dash_capstyle="round", c="k", path_effects=pe),
        dict(
            lw=1, ls=(0, (0.1, 2)), dash_capstyle="round", c="#de425b", path_effects=pe
        ),
    ]

    resolutions = ["-8", "+1", "+8"]
    for zorder, (resolution, linestyle) in enumerate(zip(resolutions, linestyles[:3])):

        try:
            x, y = get_data("m500", f"VR2915_{resolution}res_ref")
            axes[0, 0].plot(x + 1, y, **linestyle, zorder=zorder + 1)

            x, y = get_data("specific_sfr", f"VR2915_{resolution}res_ref")
            y[y < 1e-5] = np.nan
            axes[1, 0].plot(x + 1, y, **linestyle, zorder=zorder + 1)

            x, y = get_data("m500", f"VR18_{resolution}res_ref")
            axes[0, 1].plot(x + 1, y, **linestyle, zorder=zorder + 1)

            x, y = get_data("specific_sfr", f"VR18_{resolution}res_ref")
            y[y < 1e-5] = np.nan
            axes[1, 1].plot(x + 1, y, **linestyle, zorder=zorder + 1)

            x, y = get_data("bh_mass_by_id", f"VR2915_{resolution}res_ref")
            y = np.clip(y, 1e4, 1e20)
            axes[2, 0].plot(x + 1, y, **linestyle, zorder=zorder + 1)

            x, y = get_data("bh_edd_fraction_by_id", f"VR2915_{resolution}res_ref")
            y = 10 ** savgol_filter(np.log10(y), 10, 3)
            ax_accretion_group.plot(
                x + 1, y, **linestyles_mdot[zorder], zorder=zorder + 1
            )

            x, y = get_data("bh_mass_by_id", f"VR18_{resolution}res_ref")
            y = np.clip(y, 1e4, 1e20)
            axes[2, 1].plot(x + 1, y, **linestyle, zorder=zorder + 1)

            x, y = get_data("bh_edd_fraction_by_id", f"VR18_{resolution}res_ref")
            y = 10 ** savgol_filter(np.log10(y), 10, 3)
            ax_accretion_cluster.plot(
                x + 1, y, **linestyles_mdot[zorder], zorder=zorder + 1
            )

            # Mass difference gives out rubbish
            # time_diffs = []
            # for z1, z2 in zip(x[1:], x[:-1]):
            #     time_diffs.append(
            #         time_diff_from_redshift_diff(z2, z1)
            #     )
            # time_diffs = np.asarray(time_diffs)
            # mass_diffs = np.diff(y)
            # mdot_edd = 0.5 * (y[1:] + y[:-1]) / 1e8 * 2.218
            # accretion = mass_diffs / time_diffs / 1e9 / mdot_edd
            # accretion[accretion < 1e-7] =  1e-7
            # y = 10 ** savgol_filter(np.log10(accretion), 10, 3)
            # ax_accretion_cluster.plot(x[1:] + 1, y, **linestyles_mdot[zorder], zorder=zorder + 1)

        except:
            print(f"Problem with {resolution}")

    for ax in axes.flat:
        ax.autoscale(enable=True, axis="y", tight=False)
        ax.set_xlim(None, 11)
        ax.minorticks_on()

    handles = [
        lines.Line2D([], [], **linestyles[0], label="Low-res"),
        lines.Line2D([], [], **linestyles[1], label="Mid-res"),
        lines.Line2D([], [], **linestyles[2], label="High-res"),
    ]
    axes[0, 0].legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )
    axes[0, 1].legend(
        handles=handles[:-1],
        loc="upper right",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )

    handles = [lines.Line2D([], [], **linestyles_mdot[1], label="Mass-accretion rate")]
    axes[2, 0].legend(
        handles=handles,
        loc="upper center",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )
    handles = [
        lines.Line2D(
            [], [], c="grey", lw=2, ls="--", label=r"$1/t_{\rm H}$ ($0.5<z<7.5$)"
        ),
        lines.Line2D(
            [], [], c="grey", lw=2, ls="-.", label=r"$1/3t_{\rm H}$ ($0.5<z<6.5$)"
        ),
        lines.Line2D(
            [],
            [],
            c="grey",
            lw=0,
            marker="v",
            markersize=5,
            label=r"$10^{-2}$ Gyr$^{-1}$  ($z=0$)",
        ),
    ]
    axes[1, 1].legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=4,
    )

    for ax in [ax_accretion_group, ax_accretion_cluster]:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.axhspan(1e-3, 1e-1, xmin=0.925, fc="orange", ec="none", alpha=0.5)
        ax.text(
            0.925 / 2 + 0.5,
            1e-2,
            "Radio",
            rotation=90,
            ha="center",
            va="center",
            fontsize="small",
            transform=trans,
        )
        ax.axhspan(1e-1, 1e0, xmin=0.925, fc="orange", ec="none", alpha=0.75)
        ax.text(
            0.925 / 2 + 0.5,
            10 ** (-0.5),
            "Quasar",
            rotation=90,
            ha="center",
            va="center",
            fontsize="small",
            transform=trans,
        )
        ax.axhspan(1e0, 6e1, xmin=0.925, fc="orange", ec="none", alpha=1)
        ax.text(
            0.925 / 2 + 0.5,
            10 ** (0.5),
            "SE",
            rotation=90,
            ha="center",
            va="center",
            fontsize="small",
            transform=trans,
        )
        ax.axhline(1, c="orange", alpha=0.8)
        ax.axhline(1.0e-1, c="orange", alpha=0.8)

    # fig.savefig("time_evolution_1.pdf")
    plt.show()
