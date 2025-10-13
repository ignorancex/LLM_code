import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib import lines
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from zbands import z_bands_group, z_bands_cluster

plt.style.use("mnras.mplstyle")


def make_title(axes, title, padding=3, **kwargs):
    axes.annotate(
        text=title,
        xy=(0.5, 0.99),
        xytext=(padding - 1, -(padding - 1)),
        textcoords="offset pixels",
        xycoords="axes fraction",
        bbox=dict(facecolor="w", edgecolor="none", alpha=0.85, zorder=0, pad=padding),
        color="k",
        ha="center",
        va="top",
        alpha=1,
        **kwargs,
    )


def make_title_top_left(axes, title, facecolor="k"):
    txt = AnchoredText(
        title,
        loc="upper left",
        frameon=True,
        pad=0.4,
        borderpad=0,
        prop=dict(color="w", alpha=1),
    )
    txt.patch.set_edgecolor("none")
    txt.patch.set_facecolor(facecolor)
    txt.patch.set_alpha(1)
    txt.zorder = 100
    axes.add_artist(txt)


def profiles_panels_3x2(titles_on: bool = True, **kwargs):
    fig, axes_all = plt.subplots(
        2, 3, figsize=(3.3 * 2, 3.0 * 1.5), constrained_layout=True, **kwargs
    )

    for ax in axes_all.flat:
        ax.loglog()
        ax.grid(color="grey", which="major", alpha=0.3, lw=0.3, ls="--", zorder=0)

    for i, phase in enumerate("ABC"):
        if titles_on:
            axes_all[0, i].set_title(f"Phase {phase:s}")
        axes_all[1, i].set_xlabel(r"$r/r_{500}$")

    axes_all[0, 0].set_ylabel(r"$K/K_{500}$")
    axes_all[1, 0].set_ylabel(r"$K/K_{500}$")

    make_title_top_left(axes_all[0, 0], "Galaxy", facecolor="grey")
    make_title_top_left(axes_all[0, 1], "Galaxy", facecolor="grey")
    make_title_top_left(axes_all[0, 2], "Group", facecolor="black")

    make_title_top_left(axes_all[1, 0], "Galaxy", facecolor="grey")
    make_title_top_left(axes_all[1, 1], "Group", facecolor="grey")
    make_title_top_left(axes_all[1, 2], "Cluster", facecolor="black")

    return fig, axes_all


def label_line(ax, line, label, x, y, color="0.5", size=12):
    """
    Add a label to a line, at the proper angle.

    Arguments
    ---------
    line : matplotlib.lines.Line2D object,
    label : str
    x : float
        x-position to place center of text (in data coordinated
    y : float
        y-position to place center of text (in data coordinates)
    color : str
    size : float
    """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    text = ax.annotate(
        label,
        xy=(x, y),
        xytext=(-10, 0),
        textcoords="offset points",
        size=size,
        color=color,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = sp2[1] - sp1[1]
    run = sp2[0] - sp1[0]

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)
    return text


def get_data(filename: str, run: str) -> tuple:
    try:
        data = np.load(f"{runs_location}/{run}/{filename}.npy")
        radial_bin_centers = np.load(f"{runs_location}/{run}/radial_bin_centers.npy")
    except:
        print(f"problem with {runs_location}/{run}/{filename}.npy")
        return np.zeros(1), np.zeros(1)

    return radial_bin_centers, data


pe = [
    mpe.Stroke(linewidth=2.5, foreground="w"),
    mpe.Stroke(foreground="w", alpha=0.8),
    mpe.Normal(),
]

linestyles = [
    dict(lw=2, path_effects=pe),
    dict(lw=2, path_effects=pe),
    dict(lw=2, path_effects=pe),
]

color_phases = [
    ("#279EFF", "#191D88", "#2A4353"),
    ("#FE7BE5", "#9F00FF", "#7a5195"),
    ("#FFB000", "#F86F03", "#B8621B"),
]

runs_location = "../data"

fig, axes = profiles_panels_3x2(sharey="row")

for i, zband in enumerate(z_bands_group):
    object_name = "VR2915"
    j = 0

    handles = [
        lines.Line2D(
            [], [], color=color_phases[i][0], **linestyles[0], label="Low-res"
        ),
        lines.Line2D(
            [], [], color=color_phases[i][1], **linestyles[1], label="Mid-res"
        ),
        lines.Line2D(
            [], [], color=color_phases[i][2], **linestyles[2], label="High-res"
        ),
    ]
    axes[0, i].legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )
    axes[1, i].legend(
        handles=handles[:-1],
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )

    for r_id, resolution in enumerate(["-8", "+1", "+8"]):
        object_key = f"{object_name:s}_{resolution:s}res"
        redshifts = np.load(f"{runs_location}/{object_key}_ref/redshift.npy")
        time_mask = np.where((redshifts > zband.z_min) & (redshifts < zband.z_max))[0]

        try:
            radial_bin_centre, entropy_profiles = get_data(
                "entropy_profiles", f"{object_name}_{resolution}res_ref"
            )
            entropy_profiles = (
                entropy_profiles[time_mask]
                / get_data("k500", f"{object_name}_{resolution}res_ref")[1][
                    time_mask, None
                ]
            )
            entropy_profile = np.nanmedian(entropy_profiles, axis=0)
            entropy_profile_min = np.nanpercentile(entropy_profiles, 25, axis=0)
            entropy_profile_max = np.nanpercentile(entropy_profiles, 75, axis=0)
            axes[j, i].fill_between(
                radial_bin_centre,
                entropy_profile_min,
                entropy_profile_max,
                facecolor=color_phases[i][r_id],
                alpha=0.25,
                edgecolor="none",
            )
            axes[j, i].plot(
                radial_bin_centre,
                entropy_profile,
                color=color_phases[i][r_id],
                **linestyles[r_id],
            )
        except:
            pass

        median_mass = np.nanmedian(
            get_data("m500", f"{object_name}_+1res_ref")[1][time_mask]
        )
        exp = np.floor(np.log10(median_mass))
        mantissa = np.power(10, np.log10(median_mass) - exp)
        quart1_mass = np.nanmin(
            get_data("m500", f"{object_name}_+1res_ref")[1][time_mask]
        )
        quart1_mantissa = np.power(10, np.log10(quart1_mass) - exp)
        quart3_mass = np.nanmax(
            get_data("m500", f"{object_name}_+1res_ref")[1][time_mask]
        )
        quart3_mantissa = np.power(10, np.log10(quart3_mass) - exp)
        quart1_mantissa_delta = np.abs(quart1_mantissa - mantissa)
        quart3_mantissa_delta = np.abs(quart3_mantissa - mantissa)
        mass_string = (
            f"$M_{{500}}="
            f"{mantissa:.2f}^{{+{quart3_mantissa_delta:.2f}}}_{{-{quart1_mantissa_delta:.2f}}}"
            f"\\times 10^{{{exp:.0f}}}$ M$_{{\\odot}}$"
        )

        txt = AnchoredText(
            mass_string,
            loc="upper center",
            frameon=True,
            pad=0.4,
            borderpad=0.1,
            prop=dict(color="k", alpha=1, fontsize="small"),
        )
        txt.patch.set_edgecolor("none")
        txt.patch.set_facecolor("w")
        txt.patch.set_alpha(0.85)
        txt.zorder = 90
        axes[j, i].add_artist(txt)

for i, zband in enumerate(z_bands_cluster):

    j = 1
    object_name = "VR18"

    handles = [
        lines.Line2D(
            [], [], color=color_phases[i][0], **linestyles[0], label="Low-res"
        ),
        lines.Line2D(
            [], [], color=color_phases[i][1], **linestyles[1], label="Mid-res"
        ),
        lines.Line2D(
            [], [], color=color_phases[i][2], **linestyles[2], label="High-res"
        ),
    ]
    axes[0, i].legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )
    axes[1, i].legend(
        handles=handles[:-1],
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="w",
        edgecolor="none",
        handlelength=2,
    )

    for r_id, resolution in enumerate(["-8", "+1"]):
        object_key = f"{object_name:s}_{resolution:s}res"
        redshifts = np.load(f"{runs_location}/{object_key}_ref/redshift.npy")
        time_mask = np.where((redshifts > zband.z_min) & (redshifts < zband.z_max))[0]

        try:
            radial_bin_centre, entropy_profiles = get_data(
                "entropy_profiles", f"{object_name}_{resolution}res_ref"
            )
            entropy_profiles = (
                entropy_profiles[time_mask]
                / get_data("k500", f"{object_name}_{resolution}res_ref")[1][
                    time_mask, None
                ]
            )
            entropy_profile = np.nanmedian(entropy_profiles, axis=0)
            entropy_profile_min = np.nanpercentile(entropy_profiles, 25, axis=0)
            entropy_profile_max = np.nanpercentile(entropy_profiles, 75, axis=0)
            axes[j, i].fill_between(
                radial_bin_centre,
                entropy_profile_min,
                entropy_profile_max,
                facecolor=color_phases[i][r_id],
                alpha=0.25,
                edgecolor="none",
            )
            axes[j, i].plot(
                radial_bin_centre,
                entropy_profile,
                color=color_phases[i][r_id],
                **linestyles[r_id],
            )
        except:
            pass

        median_mass = np.nanmedian(
            get_data("m500", f"{object_name}_+1res_ref")[1][time_mask]
        )
        exp = np.floor(np.log10(median_mass))
        mantissa = np.power(10, np.log10(median_mass) - exp)
        quart1_mass = np.nanmin(
            get_data("m500", f"{object_name}_+1res_ref")[1][time_mask]
        )
        quart1_mantissa = np.power(10, np.log10(quart1_mass) - exp)
        quart3_mass = np.nanmax(
            get_data("m500", f"{object_name}_+1res_ref")[1][time_mask]
        )
        quart3_mantissa = np.power(10, np.log10(quart3_mass) - exp)
        quart1_mantissa_delta = np.abs(quart1_mantissa - mantissa)
        quart3_mantissa_delta = np.abs(quart3_mantissa - mantissa)
        mass_string = (
            f"$M_{{500}}="
            f"{mantissa:.2f}^{{+{quart3_mantissa_delta:.2f}}}_{{-{quart1_mantissa_delta:.2f}}}"
            f"\\times 10^{{{exp:.0f}}}$ M$_{{\\odot}}$"
        )

        txt = AnchoredText(
            mass_string,
            loc="upper center",
            frameon=True,
            pad=0.4,
            borderpad=0.1,
            prop=dict(color="k", alpha=1, fontsize="small"),
        )
        txt.patch.set_edgecolor("none")
        txt.patch.set_facecolor("w")
        txt.patch.set_alpha(0.85)
        txt.zorder = 90
        axes[j, i].add_artist(txt)

axes[0, 0].set_ylim(0.11, 22)
axes[1, 0].set_ylim(0.07, 8)

for ax in axes.flat:
    # Plot VKB line for adiabatic simulations
    r = np.array([0.01, 2.5])
    k = 1.40 * r**1.1
    vkb_line = ax.plot(r, k, c="#488f31", ls="-.", lw=1.5, zorder=1)
    ax.axvline(0.15, c="k", ls="--", zorder=0)
    ax.axvline(1.0, c="k", ls="--", zorder=0)

for i in range(2):
    label_line(
        axes[i, 1],
        vkb_line[0],
        "VKB (2005)",
        0.35,
        1.40 * 0.35**1.1,
        size="small",
        color="#488f31",
    )
    axes[i, 0].text(
        0.15 * 0.95, 0.2, "Core", fontsize="small", rotation=90, ha="right", va="bottom"
    )

axes[0, 2].text(
    0.15 * 0.95, 0.2, "Core", fontsize="small", rotation=90, ha="right", va="bottom"
)
axes[1, 2].text(
    0.15 * 0.95, 1, "Core", fontsize="small", rotation=90, ha="right", va="bottom"
)

p = Rectangle(
    (0.15, 0.16653), -0.149, 0.2897 - 0.16653, linewidth=0, fill=None, hatch="/////"
)
axes[1, 2].add_patch(p)
axes[1, 2].annotate(
    "Non-radiative plateau",
    xy=(0.025, (0.5 * np.log10(0.2897) * np.log10(0.154))),
    fontsize="small",
    rotation=0,
    ha="center",
    va="center",
    bbox=dict(facecolor="w", edgecolor="none", boxstyle="round", alpha=0.9),
)


plt.savefig("entropy_profiles.pdf")
plt.show()
