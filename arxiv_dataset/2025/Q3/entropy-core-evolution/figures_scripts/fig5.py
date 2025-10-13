import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib import lines
from matplotlib.offsetbox import AnchoredText

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


from scipy.optimize import curve_fit


def f(x, A, B):  # this is your 'straight line' y=f(x)
    return A * x + B


def get_slope(object_name, smooth):
    redshifts = np.load(f"{runs_location}/{object_name}_ref/redshift.npy")

    radial_bin_centre, entropy_profiles = get_data(
        "entropy_profiles", f"{object_name}_ref"
    )
    entropy_profiles = (
        entropy_profiles[:] / get_data("k500", f"{object_name}_ref")[1][:, None]
    )
    mass = get_data("m500", f"{object_name}_ref")[1][:]

    radius_mask_1 = np.where((radial_bin_centre > 0.04) & (radial_bin_centre < 0.15))[0]
    radius_mask_2 = np.where((radial_bin_centre > 0.5) & (radial_bin_centre < 2))[0]

    alpha1 = []
    alpha2 = []
    alpha_err1 = []
    alpha_err2 = []
    mass_scaled = mass

    for i in range(len(entropy_profiles) - smooth):
        entropy_profiles[i, :] = np.nanmean(entropy_profiles[i : i + smooth, :], axis=0)
        mass_scaled[i] = np.nanmedian(mass_scaled[i : i + smooth])

    for k in entropy_profiles:

        x = np.log10(radial_bin_centre[radius_mask_1])
        y = np.log10(k[radius_mask_1])
        nan_mask = np.where(~np.isnan(y))[0]
        x = x[nan_mask]
        y = y[nan_mask]

        if len(y) < 2:
            alpha1.append(np.nan)
            alpha_err1.append(0)
        else:
            popt, pcov = curve_fit(f, x, y)
            slope, intercept = popt
            alpha1.append(slope)
            alpha_err1.append(np.sqrt(pcov[0, 0]))

        x = np.log10(radial_bin_centre[radius_mask_2])
        y = np.log10(k[radius_mask_2])
        nan_mask = np.where(~np.isnan(y))[0]
        x = x[nan_mask]
        y = y[nan_mask]

        if len(y) < 2:
            alpha2.append(np.nan)
            alpha_err2.append(0)
        else:
            popt, pcov = curve_fit(f, x, y)
            slope, intercept = popt
            alpha2.append(slope)
            alpha_err2.append(np.sqrt(pcov[0, 0]))

    return (
        redshifts,
        mass_scaled,
        np.asarray(alpha1),
        np.asarray(alpha2),
        np.asarray(alpha_err1),
        np.asarray(alpha_err2),
    )


color_phases = [
    ("#279EFF", "#191D88", "#2A4353"),
    ("#FE7BE5", "#9F00FF", "#7a5195"),
    ("#FFB000", "#F86F03", "#B8621B"),
]

fig, (ax, ax1) = plt.subplots(
    1, 2, figsize=(6, 3), constrained_layout=True, sharey=True
)

# Plot VKB line for adiabatic simulations
ax.text(
    6e11,
    1.12,
    "VKB (2005)",
    fontsize="small",
    color="#488f31",
    rotation=0,
    ha="left",
    va="bottom",
)
ax.set_xscale("log")
ax1.set_xscale("log")
ax.set_ylim(-0.7, 1.7)
ax.set_xlim(5e11, 1.2e13)
ax1.set_xlim(2.8e12, 5e14)
ax.grid(color="grey", which="major", alpha=0.3, lw=0.3, ls="--", zorder=0)
ax1.grid(color="grey", which="major", alpha=0.3, lw=0.3, ls="--", zorder=0)
ax.set_ylabel(r"$\left\langle\alpha_{K}\right\rangle$")
ax.set_xlabel(r"$M_{500}(z)$ [M$_\odot$]")
ax1.set_xlabel(r"$M_{500}(z)$ [M$_\odot$]")
# make_title_top_left(ax, 'High-res')
# make_title_top_left(ax1, 'Mid-res')

handles = [
    lines.Line2D([], [], color="#191D88", marker="D", ms=4.5, lw=0, label="Core"),
    lines.Line2D(
        [],
        [],
        mec="#de425b",
        mfc="w",
        mew=1.5,
        marker="D",
        ms=4.5,
        lw=0,
        label="Outskirts",
    ),
]
ax1.legend(
    handles=handles,
    loc="upper left",
    frameon=True,
    framealpha=1.0,
    facecolor="w",
    edgecolor="none",
    handlelength=1,
)

handles = [
    lines.Line2D(
        [], [], color=color_phases[0][0], marker=".", ms=10, lw=0, label="Phase A"
    ),
    lines.Line2D(
        [], [], color=color_phases[1][0], marker=".", ms=10, lw=0, label="Phase B"
    ),
    lines.Line2D(
        [], [], color=color_phases[2][0], marker=".", ms=10, lw=0, label="Phase C"
    ),
    lines.Line2D([], [], color="k", marker=".", ms=10, lw=0, label="Outside phases"),
]
ax.legend(
    handles=handles,
    loc="upper left",
    frameon=True,
    framealpha=1.0,
    facecolor="w",
    ncol=2,
    edgecolor="none",
    handlelength=1,
)


object_name = "VR2915_+8res"
axis = ax
z_ranges = z_bands_group

axis.axhline(1.1, c="#488f31", ls="-.", lw=1.5, zorder=0)
axis.axhline(0, c="k", ls="-", lw=1.5, zorder=0)

masses_connect = []
slopes_connect_1 = []
slopes_connect_2 = []

z, m, a1, a2, e1, e2 = get_slope(object_name, 7)
time_mask = np.where((z_bands_group[0].z_max < z) & (z < 3.1))[0]
axis.scatter(m[time_mask], a1[time_mask], fc="k", ec="w", lw=0.3, s=6)
axis.scatter(m[time_mask], a2[time_mask], ec="k", fc="w", lw=0.5, s=6)

for i, z_range in enumerate(z_ranges):

    z, m, a1, a2, e1, e2 = get_slope(object_name, [7, 3, 2][i])

    if i in [0, 1]:
        time_mask = np.where((z < z_range.z_min) & (z > z_bands_group[i + 1].z_max))[0]
        axis.scatter(m[time_mask], a1[time_mask], fc="k", ec="w", lw=0.3, s=6)
        axis.scatter(m[time_mask], a2[time_mask], ec="k", fc="w", lw=0.5, s=6)

    redshifts = np.load(f"{runs_location}/{object_name}_ref/redshift.npy")
    time_mask = np.where((redshifts > z_range.z_min) & (redshifts < z_range.z_max))[0]

    # mass_normalized = get_data('m500', f'{object_name}_ref')[1] / get_data('m500', f'{object_name}_ref')[1][-1]
    # for l in range(len(mass_normalized)-4):
    #     mass_normalized[l] = np.nanmedian(mass_normalized[l:l+4])
    mass_normalized = m[time_mask]
    median_mass = np.nanmedian(mass_normalized)
    max_mass = np.nanmax(mass_normalized)
    min_mass = np.nanmin(mass_normalized)

    axis.plot(
        [min_mass, max_mass],
        [-0.6 - 0.1 * ((i + 1) % 2 - 1)] * 2,
        color=color_phases[i][0],
        lw=3,
        ms=0,
        zorder=4,
    )
    axis.scatter(
        [median_mass],
        [-0.6 - 0.1 * ((i + 1) % 2 - 1)],
        fc=color_phases[i][0],
        ec="none",
        s=70,
        marker="s",
        zorder=4,
    )
    axis.text(
        median_mass,
        -0.6 - 0.1 * ((i + 1) % 2 - 1),
        "ABC"[i],
        color="w",
        fontsize="small",
        ha="center",
        va="center_baseline",
        zorder=5,
    )

    if "18" in object_name and i == 2:
        axis.text(
            median_mass,
            -0.6 - 0.1 * ((i + 1) % 2 - 1) + 0.08,
            "Median",
            color="k",
            fontsize="small",
            ha="center",
            va="bottom",
        )
        axis.text(
            min_mass * 0.93,
            -0.6 - 0.1 * ((i + 1) % 2 - 1),
            "Min",
            color="k",
            fontsize="small",
            ha="right",
            va="center",
        )
        axis.text(
            max_mass * 1.07,
            -0.6 - 0.1 * ((i + 1) % 2 - 1),
            "Max",
            color="k",
            fontsize="small",
            ha="left",
            va="center",
        )

    axis.scatter(
        m[time_mask], a1[time_mask], fc=color_phases[i][0], ec="w", lw=0.3, s=6
    )
    axis.scatter(
        m[time_mask], a2[time_mask], ec=color_phases[i][0], fc="w", lw=0.5, s=6
    )

    radial_bin_centre, entropy_profiles = get_data(
        "entropy_profiles", f"{object_name}_ref"
    )
    entropy_profiles = (
        entropy_profiles[time_mask]
        / get_data("k500", f"{object_name}_ref")[1][time_mask, None]
    )
    entropy_profile = np.nanmedian(entropy_profiles, axis=0)
    radius_mask_1 = np.where((radial_bin_centre > 0.04) & (radial_bin_centre < 0.15))[0]
    radius_mask_2 = np.where((radial_bin_centre > 0.5) & (radial_bin_centre < 2))[0]

    popt, pcov = curve_fit(
        f,
        np.log10(radial_bin_centre[radius_mask_2]),
        np.log10(entropy_profile[radius_mask_2]),
    )
    median_slope = popt[0]
    max_slope = np.nanpercentile(a2[time_mask], 75)
    min_slope = np.nanpercentile(a2[time_mask], 25)
    axis.plot(
        [median_mass] * 2, [min_slope, max_slope], color="#de425b", lw=2, ms=0, zorder=2
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a2[time_mask], 50)],
        marker="D",
        fc="w",
        ec="none",
        lw=2,
        s=15,
        zorder=3,
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a2[time_mask], 50)],
        marker="D",
        fc="#de425b",
        ec="none",
        lw=2,
        s=55,
        zorder=2,
    )
    slopes_connect_2.append(np.nanpercentile(a2[time_mask], 50))
    print(median_mass, max_slope)
    popt, pcov = curve_fit(
        f,
        np.log10(radial_bin_centre[radius_mask_1]),
        np.log10(entropy_profile[radius_mask_1]),
    )
    median_slope = popt[0]
    max_slope = np.nanpercentile(a1[time_mask], 75)
    min_slope = np.nanpercentile(a1[time_mask], 25)
    axis.plot(
        [median_mass] * 2, [min_slope, max_slope], color="#191D88", lw=2, ms=0, zorder=4
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a1[time_mask], 50)],
        marker="D",
        fc="w",
        ec="none",
        lw=2,
        s=80,
        zorder=4,
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a1[time_mask], 50)],
        marker="D",
        fc="#191D88",
        ec="none",
        lw=2,
        s=40,
        zorder=5,
    )
    masses_connect.append(median_mass)
    slopes_connect_1.append(np.nanpercentile(a1[time_mask], 50))

axis.plot(masses_connect, slopes_connect_2, color="#de425b", lw=0, ms=0, zorder=1)
axis.plot(masses_connect, slopes_connect_1, color="#191D88", lw=0, ms=0, zorder=3)

object_name = "VR18_+1res"
axis = ax1
z_ranges = z_bands_cluster

axis.axhline(1.1, c="#488f31", ls="-.", lw=1.5, zorder=0)
axis.axhline(0, c="k", ls="-", lw=1.5, zorder=0)

masses_connect = []
slopes_connect_1 = []
slopes_connect_2 = []

z, m, a1, a2, e1, e2 = get_slope(object_name, 7)
time_mask = np.where((z_bands_cluster[0].z_max < z) & (z < 4.3))[0]
axis.scatter(m[time_mask], a1[time_mask], fc="k", ec="w", lw=0.3, s=6)
axis.scatter(m[time_mask], a2[time_mask], ec="k", fc="w", lw=0.5, s=6)

for i, z_range in enumerate(z_ranges):

    z, m, a1, a2, e1, e2 = get_slope(object_name, [7, 3, 2][i])

    if i in [0, 1]:
        time_mask = np.where((z < z_range.z_min) & (z > z_bands_cluster[i + 1].z_max))[
            0
        ]
        axis.scatter(m[time_mask], a1[time_mask], fc="k", ec="w", lw=0.3, s=6)
        axis.scatter(m[time_mask], a2[time_mask], ec="k", fc="w", lw=0.5, s=6)

    redshifts = np.load(f"{runs_location}/{object_name}_ref/redshift.npy")
    time_mask = np.where((redshifts > z_range.z_min) & (redshifts < z_range.z_max))[0]

    # mass_normalized = get_data('m500', f'{object_name}_ref')[1] / get_data('m500', f'{object_name}_ref')[1][-1]
    # for l in range(len(mass_normalized)-4):
    #     mass_normalized[l] = np.nanmedian(mass_normalized[l:l+4])
    mass_normalized = m[time_mask]
    median_mass = np.nanmedian(mass_normalized)
    max_mass = np.nanmax(mass_normalized)
    min_mass = np.nanmin(mass_normalized)

    axis.plot(
        [min_mass, max_mass],
        [-0.6 - 0.1 * ((i + 1) % 2 - 1)] * 2,
        color=color_phases[i][0],
        lw=3,
        ms=0,
        zorder=4,
    )
    axis.scatter(
        [median_mass],
        [-0.6 - 0.1 * ((i + 1) % 2 - 1)],
        fc=color_phases[i][0],
        ec="none",
        s=70,
        marker="s",
        zorder=4,
    )
    axis.text(
        median_mass,
        -0.6 - 0.1 * ((i + 1) % 2 - 1),
        "ABC"[i],
        color="w",
        fontsize="small",
        ha="center",
        va="center_baseline",
        zorder=5,
    )

    if "18" in object_name and i == 2:
        axis.text(
            median_mass,
            -0.6 - 0.1 * ((i + 1) % 2 - 1) + 0.08,
            "Median",
            color="k",
            fontsize="small",
            ha="center",
            va="bottom",
        )
        axis.text(
            min_mass * 0.93,
            -0.6 - 0.1 * ((i + 1) % 2 - 1),
            "Min",
            color="k",
            fontsize="small",
            ha="right",
            va="center",
        )
        axis.text(
            max_mass * 1.07,
            -0.6 - 0.1 * ((i + 1) % 2 - 1),
            "Max",
            color="k",
            fontsize="small",
            ha="left",
            va="center",
        )

    axis.scatter(
        m[time_mask], a1[time_mask], fc=color_phases[i][0], ec="w", lw=0.3, s=6
    )
    axis.scatter(
        m[time_mask], a2[time_mask], ec=color_phases[i][0], fc="w", lw=0.5, s=6
    )

    radial_bin_centre, entropy_profiles = get_data(
        "entropy_profiles", f"{object_name}_ref"
    )
    entropy_profiles = (
        entropy_profiles[time_mask]
        / get_data("k500", f"{object_name}_ref")[1][time_mask, None]
    )
    entropy_profile = np.nanmedian(entropy_profiles, axis=0)
    radius_mask_1 = np.where((radial_bin_centre > 0.04) & (radial_bin_centre < 0.15))[0]
    radius_mask_2 = np.where((radial_bin_centre > 0.5) & (radial_bin_centre < 2))[0]

    popt, pcov = curve_fit(
        f,
        np.log10(radial_bin_centre[radius_mask_2]),
        np.log10(entropy_profile[radius_mask_2]),
    )
    median_slope = popt[0]
    max_slope = np.nanpercentile(a2[time_mask], 75)
    min_slope = np.nanpercentile(a2[time_mask], 25)
    axis.plot(
        [median_mass] * 2, [min_slope, max_slope], color="#de425b", lw=2, ms=0, zorder=2
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a2[time_mask], 50)],
        marker="D",
        fc="w",
        ec="none",
        lw=2,
        s=15,
        zorder=3,
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a2[time_mask], 50)],
        marker="D",
        fc="#de425b",
        ec="none",
        lw=2,
        s=55,
        zorder=2,
    )
    slopes_connect_2.append(np.nanpercentile(a2[time_mask], 50))
    print(median_mass, max_slope)

    popt, pcov = curve_fit(
        f,
        np.log10(radial_bin_centre[radius_mask_1]),
        np.log10(entropy_profile[radius_mask_1]),
    )
    median_slope = popt[0]
    max_slope = np.nanpercentile(a1[time_mask], 75)
    min_slope = np.nanpercentile(a1[time_mask], 25)
    axis.plot(
        [median_mass] * 2, [min_slope, max_slope], color="#191D88", lw=2, ms=0, zorder=2
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a1[time_mask], 50)],
        marker="D",
        fc="w",
        ec="none",
        lw=2,
        s=80,
        zorder=2,
    )
    axis.scatter(
        [median_mass],
        [np.nanpercentile(a1[time_mask], 50)],
        marker="D",
        fc="#191D88",
        ec="none",
        lw=2,
        s=40,
        zorder=3,
    )

    masses_connect.append(median_mass)
    slopes_connect_1.append(np.nanpercentile(a1[time_mask], 50))

axis.plot(masses_connect, slopes_connect_2, color="#de425b", lw=0, ms=0, zorder=1)
axis.plot(masses_connect, slopes_connect_1, color="#191D88", lw=0, ms=0, zorder=3)

ax.annotate(
    "Galaxy",
    xy=(1396256485172.1714, 0.9486175991555141),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    weight="bold",
    color="k",
    bbox=dict(facecolor="w", edgecolor="none", boxstyle="round", alpha=0.8),
)
ax.annotate(
    "Galaxy",
    xy=(3791300992245.0317, 1.0248750584227018),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    weight="bold",
    color="k",
    bbox=dict(facecolor="w", edgecolor="none", boxstyle="round", alpha=0.8),
)
ax.annotate(
    "Group",
    xy=(7966171912082.73, 0.24565554969835285),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    weight="bold",
    color="w",
    bbox=dict(facecolor="k", edgecolor="none", boxstyle="round", alpha=0.8),
)
ax1.annotate(
    "Galaxy",
    xy=(5530865830138.316, 0.8679411005149356),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    weight="bold",
    color="k",
    bbox=dict(facecolor="w", edgecolor="none", boxstyle="round", alpha=0.8),
)
ax1.annotate(
    "Group",
    xy=(39417039725509.03, 0.9141268338478978),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    weight="bold",
    color="k",
    bbox=dict(facecolor="w", edgecolor="none", boxstyle="round", alpha=0.8),
)
ax1.annotate(
    "Cluster",
    xy=(230716668092951.28, 1.013974664926522),
    xytext=(0, 14),
    textcoords="offset points",
    ha="center",
    va="bottom",
    weight="bold",
    color="w",
    bbox=dict(facecolor="k", edgecolor="none", boxstyle="round", alpha=0.8),
)


plt.savefig("entropy_log_slope.pdf")
plt.show()
