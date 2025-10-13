import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Tuple
from matplotlib import lines, transforms
from matplotlib.ticker import ScalarFormatter
import matplotlib.patheffects as mpe

from zbands import z_bands_group, z_bands_cluster
from lookback import redshift_from_lookback_time

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


def get_data(filename: str, run: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data and redshift for the given filename and run."""
    try:
        data = np.load(os.path.join(RUNS_LOCATION, run, f"{filename}.npy"))
        redshift = np.load(os.path.join(RUNS_LOCATION, run, "redshift.npy"))
        return redshift, data
    except FileNotFoundError:
        print(f"problem with {os.path.join(RUNS_LOCATION, run, f"{filename}.npy")}")
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

        ax = axes[2, i]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)
        ax.text(
            z_band.label_midpoint_log,
            0.05,
            z_band.time_label,
            rotation=90,
            fontsize=5,
            ha="center",
            va="bottom",
            color="k",
            transform=trans,
            zorder=100,
        )
        ax = axes[1, i]
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

        ax = axes[2, i]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)
        ax.text(
            z_band.label_midpoint_log,
            0.05,
            z_band.time_label,
            rotation=90,
            fontsize=5,
            ha="center",
            va="bottom",
            color="k",
            transform=trans,
            zorder=100,
        )
        ax = axes[1, i]
        ax.axvspan(z_band.z_min + 1, z_band.z_max + 1, **shaded_kwargs)

    z_ticks = [1, 2, 3, 5, 8, 11, 16]
    for i, ax in enumerate(axes.flat):
        ax.loglog()
        ax.grid(color="grey", which="major", alpha=0.3, lw=0.3, ls="--", zorder=0)

        # Add the redsift labels
        ax.set_xticks(z_ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticklabels([rf"${z - 1:.0f}$" for z in z_ticks])
        ax.set_xlim(zp1_low, zp1_high)
        ax.minorticks_on()

    # Add the time labels
    time_axis_ticks = [0, 5, 8, 11, 13, 13.5]
    for ax in [axes[0, 0], axes[0, 1]]:
        ax1_time = ax.twiny()
        ax1_time.set_xscale("log")
        ax1_time.set_xlabel("Lookback time [Gyr]", labelpad=10)
        ax1_time.set_xlim(zp1_low, zp1_high)
        ax1_time.minorticks_off()
        ax1_time.set_xticks(
            [redshift_from_lookback_time(t) + 1 for t in time_axis_ticks]
        )
        ax1_time.set_xticklabels([f"{t:.1f}" for t in time_axis_ticks])

    for i, ylabel in enumerate(ylabels):
        axes[i, 0].set_ylabel(ylabel)

    for i in range(2):
        old_lims = axes[0, i].get_ylim()
        axes[0, i].set_ylim(old_lims[0], old_lims[1] * 3)
        axes[num_rows - 1, i].set_xlabel("Redshift")

        axes[0, i].axhline(0.157, ls="-.", c="k")

    trans = transforms.blended_transform_factory(
        axes[0, 0].transAxes, axes[0, 0].transData
    )
    axes[0, 0].text(
        0.05,
        0.165,
        r"$f_{\rm bary}=0.157$",
        ha="left",
        va="bottom",
        fontsize=8,
        transform=trans,
    )
    trans = transforms.blended_transform_factory(
        axes[0, 1].transAxes, axes[0, 1].transData
    )
    axes[0, 1].text(
        0.97, 0.165, "Planck 2018", ha="right", va="bottom", fontsize=8, transform=trans
    )

    return fig, axes


if __name__ == "__main__":

    ylabels = (
        r"$(M_{\rm gas} + M_{\star})\, /\, M_{500}$",
        r"$M_{\rm gas}\, /\, M_{500}$",
        r"$M_{\star}\, /\, M_{500}$",
    )
    fig, axes = make_time_evolution_canvas(ylabels)
    ax_accretion_group = axes[1, 0].twinx()
    ax_accretion_group.set_yscale("log")
    ax_accretion_group.set_ylim(1e-3, 20)
    ax_accretion_group.minorticks_off()
    axes[1, 0].set_zorder(ax_accretion_group.get_zorder() + 1)
    axes[1, 0].patch.set_visible(False)

    ax_accretion_cluster = axes[1, 1].twinx()
    ax_accretion_cluster.set_yscale("log")
    ax_accretion_cluster.set_ylabel("BHMAR", labelpad=10)
    ax_accretion_cluster.set_ylim(1e-3, 60)
    ax_accretion_cluster.minorticks_off()
    ax_accretion_cluster.patch.set_facecolor("none")
    axes[1, 1].set_zorder(ax_accretion_cluster.get_zorder() + 1)
    axes[1, 1].patch.set_visible(False)

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

            _, m500 = get_data("m500", f"VR2915_{resolution}res_ref")

            x, y = get_data("baryon_mass", f"VR2915_{resolution}res_ref")
            # y = 10 ** savgol_filter(np.log10(y), 10, 3)
            y[y < 1e5] = np.nan
            axes[0, 0].plot(x + 1, y / m500, **linestyle, zorder=zorder + 1)

            _, y_c = get_data("cold_gas_mass", f"VR2915_{resolution}res_ref")
            # y_c = 10 ** savgol_filter(np.log10(y_c), 10, 3)
            y_c[y - y_c < 1e7] = np.nan
            axes[0, 0].plot(
                x + 1,
                (y - y_c) / m500,
                **linestyle,
                zorder=zorder + 1,
                ls=(0, (0.1, 2)),
                dash_capstyle="round",
            )

            x, y = get_data("hot_gas_mass", f"VR2915_{resolution}res_ref")
            y[y < 1e5] = np.nan
            axes[1, 0].plot(x + 1, y / m500, **linestyle, zorder=zorder + 1)

            x, y = get_data("cold_gas_mass", f"VR2915_{resolution}res_ref")
            axes[1, 0].plot(
                x + 1,
                y / m500,
                **linestyle,
                zorder=zorder + 1,
                ls=(0, (0.1, 2)),
                dash_capstyle="round",
            )

            x, y = get_data("star_mass", f"VR2915_{resolution}res_ref")
            y = 10 ** savgol_filter(np.log10(y), 10, 3)
            axes[2, 0].plot(x + 1, y / m500, **linestyle, zorder=zorder + 1)

            x, y = get_data("bh_edd_fraction_by_id", f"VR2915_{resolution}res_ref")
            y = 10 ** savgol_filter(np.log10(y), 10, 3)
            ax_accretion_group.plot(
                x + 1, y, **linestyles_mdot[zorder], zorder=zorder + 1
            )

            _, m500 = get_data("m500", f"VR18_{resolution}res_ref")

            x, y = get_data("baryon_mass", f"VR18_{resolution}res_ref")
            # y = 10 ** savgol_filter(np.log10(y), 10, 3)
            y[y < 1e5] = np.nan
            axes[0, 1].plot(x + 1, y / m500, **linestyle, zorder=zorder + 1)

            _, y_c = get_data("cold_gas_mass", f"VR18_{resolution}res_ref")
            # y_c = 10 ** savgol_filter(np.log10(y_c), 10, 3)
            y_c[y - y_c < 1e7] = np.nan
            axes[0, 1].plot(
                x + 1,
                (y - y_c) / m500,
                **linestyle,
                zorder=zorder + 1,
                ls=(0, (0.1, 2)),
                dash_capstyle="round",
            )

            x, y = get_data("hot_gas_mass", f"VR18_{resolution}res_ref")
            y[y < 1e5] = np.nan
            axes[1, 1].plot(x + 1, y / m500, **linestyle, zorder=zorder + 1)

            x, y = get_data("cold_gas_mass", f"VR18_{resolution}res_ref")
            axes[1, 1].plot(
                x + 1,
                y / m500,
                **linestyle,
                zorder=zorder + 1,
                ls=(0, (0.1, 2)),
                dash_capstyle="round",
            )

            x, y = get_data("star_mass", f"VR18_{resolution}res_ref")
            y = 10 ** savgol_filter(np.log10(y), 10, 3)
            axes[2, 1].plot(x + 1, y / m500, **linestyle, zorder=zorder + 1)

            x, y = get_data("bh_edd_fraction_by_id", f"VR18_{resolution}res_ref")
            y = 10 ** savgol_filter(np.log10(y), 10, 3)
            ax_accretion_cluster.plot(
                x + 1, y, **linestyles_mdot[zorder], zorder=zorder + 1
            )

        except:
            pass

    for ax in axes.flat:
        ax.autoscale(enable=True, axis="y", tight=False)
        ax.minorticks_on()

    for i in range(2):
        axes[0, i].set_ylim(6e-3, 0.45)
        axes[1, i].set_ylim(3e-4, 0.45)
        axes[2, i].set_ylim(3e-3, 0.1)
        axes[0, i].set_xlim(None, 11)

    handles = [
        lines.Line2D([], [], **linestyles[0], label="Low-res"),
        lines.Line2D([], [], **linestyles[1], label="Mid-res"),
        lines.Line2D([], [], **linestyles[2], label="High-res"),
    ]
    axes[2, 0].legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )
    axes[2, 1].legend(
        handles=handles[:-1],
        loc="upper right",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )

    handles = [
        lines.Line2D([], [], lw=2, c="k", ls="-", label="With cold gas"),
        lines.Line2D(
            [],
            [],
            lw=2,
            c="k",
            ls=(0, (0.1, 2)),
            dash_capstyle="round",
            label="Without cold gas",
        ),
    ]
    axes[0, 1].legend(
        handles=handles,
        loc="center left",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=1.5,
    )

    handles = [
        lines.Line2D([], [], lw=2, c="k", ls="-", label="Hot gas"),
        lines.Line2D(
            [],
            [],
            lw=2,
            c="k",
            ls=(0, (0.1, 2)),
            dash_capstyle="round",
            label="Cold gas",
        ),
    ]
    axes[1, 0].legend(
        handles=handles,
        loc="lower center",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=1.5,
    )

    handles = [lines.Line2D([], [], **linestyles_mdot[1], label="Mass-accretion rate")]
    axes[1, 1].legend(
        handles=handles,
        loc="lower center",
        frameon=True,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )

    fig.savefig("time_evolution_2.pdf")

    plt.show()
