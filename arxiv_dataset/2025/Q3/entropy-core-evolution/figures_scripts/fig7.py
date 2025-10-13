import attr
import numpy as np
from typing import Tuple
import matplotlib.transforms as transforms

from matplotlib import pyplot as plt

plt.style.use("mnras.mplstyle")

from matplotlib.ticker import ScalarFormatter
from matplotlib import lines
from scipy.signal import savgol_filter
import matplotlib.patheffects as mpe

from lookback import redshift_from_lookback_time


def make_time_evolution_canvas(ylabels: Tuple[str]) -> Tuple[plt.Figure, plt.Axes]:
    num_rows = len(ylabels)
    num_cols = 3
    zp1_low, zp1_high = 0.9, 3.1

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6, 7),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1] * num_rows},
        sharex=True,
        sharey="row",
    )
    fig.set_facecolor("white")

    z_ticks = [1, 1.3, 2, 3]
    for i, ax in enumerate(axes.flat):
        ax.loglog()
        ax.grid(color="grey", which="major", alpha=0.3, lw=0.3, ls="--", zorder=0)

        # Add the redsift labels
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xticks(z_ticks)
        ax.set_xticklabels([rf"${z - 1:.1f}$" for z in z_ticks])
        ax.set_xlim(zp1_low, zp1_high)

    # Add the time labels
    time_axis_ticks = [0, 5, 8]
    for i in range(num_cols):
        ax1_time = axes[0, i].twiny()
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

    titles = [r"$r<0.15\, r_{500}$", r"$0.15 <r / r_{500}<1$", r"$1 <r / r_{500}<6$"]
    for i in range(num_cols):
        axes[0, i].set_title(titles[i], fontsize=11, pad=10)
        old_lims = axes[0, i].get_ylim()
        axes[0, i].set_ylim(old_lims[0], old_lims[1] * 3)

        axes[num_rows - 1, i].set_xlabel("Redshift")

    return fig, axes


runs_location = "../data"
RUN = "VR18_+1res"


def get_data(filename: str, run: str) -> tuple:
    data = np.load(f"{runs_location}/{run}/{filename}.npy")
    redshift = np.load(f"{runs_location}/{run}/redshift.npy")

    return redshift, data


@attr.s(auto_attribs=True)
class LineStyle:
    this_run_model: str
    origin_model: float
    kwargs: float
    label: str


pe = [
    mpe.Stroke(linewidth=2.5, foreground="w"),
    mpe.Stroke(foreground="w", alpha=0.6),
    mpe.Normal(),
]

linestyles = [
    LineStyle(
        this_run_model="ref",
        origin_model="ref",
        kwargs=dict(lw=2, c="red", path_effects=pe),
        label="Ref($z>0$) $\\in$ Ref($z=0$)",
    ),
    LineStyle(
        this_run_model="adiabatic",
        origin_model="nr",
        kwargs=dict(lw=2, c="blue", path_effects=pe),
        label="NR($z>0$) $\\in$ NR($z=0$)",
    ),
    LineStyle(
        this_run_model="ref",
        origin_model="nr",
        kwargs=dict(lw=2, c="red", ls="--", path_effects=pe),
        label="Ref($z>0$) $\\in$ NR($z=0$)",
    ),
    LineStyle(
        this_run_model="adiabatic",
        origin_model="ref",
        kwargs=dict(lw=2, c="blue", ls="--", path_effects=pe),
        label="NR($z>0$) $\\in$ Ref($z=0$)",
    ),
]

fig, axes = make_time_evolution_canvas(
    (
        r"$r_{\rm CoP}/r_{500}(z)$",
        r"$K/K_{500}(z=0)$",
        r"$T/T_{500}(z=0)$",
        r"$\rho/\rho_{\rm mean}(z)$",
    )
)

for i, rfield in enumerate(["core", "shell", "field"]):
    for j, quant in enumerate(["radius", "entropy", "temperature", "density"]):

        for linestyle in linestyles:

            _, data = get_data(
                f"gas_history_{quant}_R{rfield}_Thot_O{linestyle.origin_model}",
                f"{RUN}_{linestyle.this_run_model}",
            )
            redshifts = np.load(f"{runs_location}/{RUN}_ref/redshift.npy")

            if quant == "density":
                data[:-1] *= 0.3111
            elif quant == "radius":
                data[:-1] = 10 ** savgol_filter(np.log10(data[:-1]), 5, 2)

                axes[j, i].axhline(1, lw=1.2, c="k")

            try:
                axes[j, i].plot(redshifts + 1, data[:-1] / data[-1], **linestyle.kwargs)
            except:
                pass

        trans = transforms.blended_transform_factory(
            axes[0, 0].transAxes, axes[0, 0].transData
        )

        axes[0, 0].axhline(0.15, lw=1.2, c="k")
        axes[0, 0].text(
            0.95,
            0.155,
            "Core radius",
            transform=trans,
            ha="right",
            va="bottom",
            zorder=100,
            color="k",
        )

min_temperature = (
    100.0
    / np.load(
        f"{runs_location}/{RUN}_adiabatic/gas_history_temperature_Rcore_Thot_Oref.npy"
    )[-1]
)
baryon_fraction = 0.157

handles = [lines.Line2D([], [], **ls.kwargs, label=ls.label) for ls in linestyles]
axes[3, 2].legend(
    handles=handles,
    loc="upper right",
    frameon=True,
    framealpha=1,
    facecolor="w",
    edgecolor="none",
    handlelength=3,
)

for ax in axes.flat:
    ax.autoscale(enable=True, axis="y", tight=True)
    ax.set_xlim(None, 3.1)
    ax.minorticks_on()

axes[1, 0].set_ylim(7e-3, 4)
axes[2, 0].set_ylim(7e-3, 3)
axes[3, 0].set_ylim(5e-1, 4e3)
axes[0, 0].set_ylim(9e-2, 20)

axes[0, 0].text(
    0.0,
    1.5,
    f"Cluster mid-res",
    transform=axes[0, 0].transAxes,
    ha="left",
    va="bottom",
    fontsize="x-large",
    weight="bold",
    zorder=100,
    color="k",
)
plt.savefig("lagrangian_shells_cluster_midres_lowz_all.pdf")
plt.show()
