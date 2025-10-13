import colorsys
import matplotlib.colors as mc
import numpy as np
from scipy.stats import chi2
from math import log10, floor, erf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.tri as tri
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors as mpl_colors
from matplotlib.collections import PatchCollection
from scipy.interpolate import splprep, splev

import scipy

from . import const
from alp.models import ALP
from alp.gminus2 import DELTA_COMBINED_2023_SM_2025, DELTA_a_electron_LKB

###########################
#
fsize = 11
fsize_annotate = 10

std_figsize = (1.2 * 3.7, 1.3 * 2.3617)
std_axes_form = [0.18, 0.16, 0.79, 0.76]

rcparams = {
    "axes.labelsize": fsize,
    "xtick.labelsize": fsize,
    "ytick.labelsize": fsize,
    "figure.figsize": std_figsize,
    "legend.frameon": False,
    "legend.loc": "best",
}
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
matplotlib.rcParams["hatch.linewidth"] = 0.3

CB_color_cycle = [
    "#377eb8",
    "#f781bf",
    "#4daf4a",
    "#999999",
    "#ff7f00",
    "#a65628",
    "#984ea3",
    "#e41a1c",
    "#dede00",
]
CB_color_cycle_2 = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CB_color_cycle_2)
rcParams.update(rcparams)

# settings for Mini Figs
TOTAL_RATE = False
INCLUDE_MB_LAST_BIN = False
STACKED = False
PLOT_FAMILY = False
PATH_PLOTS = "plots/event_rates/"

PAPER_TAG = r"HKZ\,2024"


##########################
#
def get_CL_from_sigma(sigma):
    return erf(sigma / np.sqrt(2))


def get_chi2vals_w_nsigmas(n_sigmas, ndof):
    return [chi2.ppf(get_CL_from_sigma(i), ndof) for i in range(n_sigmas + 1)]


def get_chi2vals_w_sigma(sigma, ndof):
    return chi2.ppf(get_CL_from_sigma(sigma), ndof)


def get_chi2vals_w_CL(CLs, ndof):
    return [chi2.ppf(cl, ndof) for cl in CLs]


def std_fig(ax_form=std_axes_form, figsize=std_figsize, rasterized=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig, ax


def double_axes_fig(
    height=0.5,
    gap=0.1,
    axis_base=[0.14, 0.1, 0.80, 0.18],
    figsize=std_figsize,
    split_y=False,
    split_x=False,
    rasterized=False,
):
    fig = plt.figure(figsize=figsize)

    if split_y and not split_x:
        axis_base = [0.14, 0.1, 0.80, 0.4 - gap / 2]
        axis_appended = [0.14, 0.5 + gap / 2, 0.80, 0.4 - gap / 2]

    elif not split_y and split_x:
        axis_appended = [0.14, 0.1, 0.4 - gap / 2, 0.8]
        axis_base = [0.14 + 0.4 + gap / 2, 0.1, 0.4 - gap / 2, 0.8]

    else:
        axis_base[-1] = height
        axis_appended = axis_base + np.array(
            [0, height + gap, 0, 1 - 2 * height - gap - axis_base[1] - 0.07]
        )

    ax1 = fig.add_axes(axis_appended, rasterized=rasterized)
    ax2 = fig.add_axes(axis_base, rasterized=rasterized)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    return fig, [ax1, ax2]


def data_plot(ax, X, Y, xerr, yerr, zorder=2, label="data", **kwargs):
    return ax.errorbar(
        X,
        Y,
        yerr=yerr,
        xerr=xerr,
        marker="o",
        markeredgewidth=0.75,
        capsize=1,
        markerfacecolor="black",
        markeredgecolor="black",
        ms=1.75,
        lw=0.0,
        elinewidth=0.75,
        color="black",
        label=label,
        zorder=zorder,
        **kwargs,
    )


def step_plot(
    ax, x, y, lw=1, color="red", label="signal", where="post", dashes=(3, 0), zorder=3
):
    return ax.step(
        np.append(x, np.max(x) + x[-1]),
        np.append(y, 0.0),
        where=where,
        lw=lw,
        dashes=dashes,
        color=color,
        label=label,
        zorder=zorder,
    )


# Function to find the path that connects points in order of closest proximity
def nearest_neighbor_path(points):
    # Compute the pairwise distance between points
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    ordered_points = points[path]
    return ordered_points


def plot_closed_region(points, logx=False, logy=False):
    x, y = points
    if logy:
        if (y == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = np.sign(y)
        ssy = (np.abs(y) < 1) * (-1) + (np.abs(y) > 1) * (1)
        y = ssy * np.log(y * sy)
    if logx:
        if (x == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx = np.sign(x)
        ssx = (x < 1) * (-1) + (x > 1) * (1)
        x = ssx * np.log(x * sx)

    points = np.array([x, y]).T

    points_s = points - points.mean(0)
    angles = np.angle((points_s[:, 0] + 1j * points_s[:, 1]))
    points_sort = points_s[angles.argsort()]
    points_sort += points.mean(0)

    tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    u_new = np.linspace(u.min(), u.max(), len(points[:, 0]))
    x_new, y_new = splev(u_new, tck, der=0)

    if logx:
        x_new = sx * np.exp(ssx * x_new)
    if logy:
        y_new = sy * np.exp(ssy * y_new)

    return x_new, y_new


def get_ordered_closed_region(points, logx=False, logy=False):
    x, y = points
    # check for nans
    if np.isnan(points).sum() > 0:
        raise ValueError("NaN's were found in input data. Cannot order the contour.")

    # check for repeated x-entries --
    # this is an error because
    x, mask_diff = np.unique(x, return_index=True)
    y = y[mask_diff]

    if logy:
        if (y == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = 1  # np.sign(y)
        ssy = 1  # (np.abs(y) < 1) * (-1) + (np.abs(y) > 1) * (1)
        y = ssy * np.log10(y * sy)
    if logx:
        if (x == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx = 1  # np.sign(x)
        ssx = 1  # (x < 1) * (-1) + (x > 1) * (1)
        x = ssx * np.log10(x * sx)

    xmin, ymin = np.min(x), np.min(y)
    x, y = x - xmin, y - ymin

    points = np.array([x, y]).T
    # points_s     = (points - points.mean(0))
    # angles       = np.angle((points_s[:,0] + 1j*points_s[:,1]))
    # points_sort  = points_s[angles.argsort()]
    # points_sort += points.mean(0)

    # if np.isnan(points_sort).sum()>0:
    #     raise ValueError("NaN's were found in sorted points. Cannot order the contour.")
    # # print(points.mean(0))
    # # return points_sort
    # tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    # # u_new = np.linspace(u.min(), u.max(), len(points[:,0]))
    # x_new, y_new = splev(u, tck, der=0)
    # # x_new, y_new = splev(u_new, tck, der=0)
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    x_new, y_new = points[path].T
    x_new, y_new = x_new + xmin, y_new + ymin

    if logx:
        x_new = sx * 10 ** (ssx * x_new)
    if logy:
        y_new = sy * 10 ** (ssy * y_new)

    return x_new, y_new


def interp_grid(
    x,
    y,
    z,
    fine_gridx=False,
    fine_gridy=False,
    logx=False,
    logy=False,
    method="interpolate",
    smear_stddev=False,
):
    # default
    if not fine_gridx:
        fine_gridx = 100
    if not fine_gridy:
        fine_gridy = 100

    # log scale x
    if logx:
        xi = np.geomspace(np.min(x), np.max(x), fine_gridx)
    else:
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)

    # log scale y
    if logy:
        yi = np.geomspace(np.min(y), np.max(y), fine_gridy)
    else:
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    Xi, Yi = np.meshgrid(xi, yi)
    # if logy:
    #     Yi = 10**(-Yi)

    # triangulation
    if method == "triangulation":
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Zi = interpolator(Xi, Yi)

    elif method == "interpolate":
        Zi = scipy.interpolate.griddata(
            (x, y), z, (xi[None, :], yi[:, None]), method="linear", rescale=True
        )
    else:
        print(f"Method {method} not implemented.")

    # gaussian smear -- not recommended
    if smear_stddev:
        Zi = scipy.ndimage.filters.gaussian_filter(
            Zi, smear_stddev, mode="nearest", order=0, cval=0
        )

    return Xi, Yi, Zi


def round_sig(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def sci_notation(
    num,
    sig_digits=1,
    precision=None,
    exponent=None,
    notex=False,
    optional_sci=False,
):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num != 0:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), sig_digits)
        if coeff == 10:
            coeff = 1
            exponent += 1
        if precision is None:
            precision = sig_digits

        if optional_sci and np.abs(exponent) < optional_sci:
            string = rf"{round_sig(num, precision)}"
        else:
            string = r"{0:.{2}f}\times 10^{{{1:d}}}".format(coeff, exponent, precision)

        if notex:
            return string
        else:
            return f"${string}$"

    else:
        return r"0"


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


###########################
def get_cmap_colors(name, ncolors, cmin=0, cmax=1, reverse=False):
    try:
        cmap = plt.get_cmap(name)
    except ValueError:
        cmap = build_cmap(name, reverse=reverse)
    return cmap(np.linspace(cmin, cmax, ncolors, endpoint=True))


def build_cmap(color, reverse=False):
    cvals = [0, 1]
    colors = [color, "white"]
    if reverse:
        colors = colors[::-1]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    return mpl_colors.LinearSegmentedColormap.from_list("", tuples)


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="none",
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def main_plot_LFV(
    BP_NAME,
    c_lepton,
    c_NN,
    mN,
    fa_power=1,
    plot_DUNEs=True,
    figsize=(5, 6),
    ymax=1e-3,
    ymin=1e-10,
    xmin=1e-2,
    xmax=2,
    ncol=1,
    loc="lower left",
    yscale="log",
    xscale="log",
    legend=True,
    name_modifier="",
    vlines=True,
    title=None,
    smear_stddev=False,
    linewidth=0,
    annotate=False,
    plot_other=False,
    grid=False,
):

    fig, ax = std_fig(figsize=figsize)

    plot_other_limits(
        ax, c_lepton=c_lepton, c_NN=c_NN, mN=mN, linewidth=linewidth, annotate=annotate
    )

    # labels for legend
    labels = []
    labelnames = []
    name = BP_NAME

    Nsig = 2.3
    X, Y, Z = np.load(f"data/CHARM_rates_{name}.npy", allow_pickle=True)
    c = ax.contourf(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[0], 0.5)],
        alpha=1,
        zorder=-0.2,
    )
    _ = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("CHARM")

    X, Y, Z = np.load(f"data/BEBC_rates_{name}.npy", allow_pickle=True)
    c = ax.contourf(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[1], 0.5)],
        alpha=1,
        zorder=-0.1,
    )
    _ = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[1],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("BEBC")

    X, Y, Z = np.load(f"data/NoVA_rates_{name}.npy", allow_pickle=True)
    # c = ax.contourf(X, Y**fa_power, Z, levels=[Nsig, 1e100], colors=[lighten_color('darkorange', 0.5)], alpha=1, zorder=1.3)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[3],
        linestyles=[(1, (2, 0))],
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("NOvA")

    X, Y, Z = np.load(f"data/MicroBooNE_rates_{name}.npy", allow_pickle=True)
    # c = ax.contourf(X, Y**fa_power, Z, levels=[Nsig, 1e100], colors=[lighten_color('black', 0.5)], alpha=1, zorder=1.2)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[3],
        linestyles=[
            (
                1,
                (
                    1,
                    0.5,
                ),
            )
        ],
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"$\mu$BooNE (NuMI)")

    X, Y, Z = np.load(f"data/ICARUS_rates_{name}.npy", allow_pickle=True)
    # c = ax.contourf(X, Y**fa_power, Z, levels=[Nsig, 1e100], colors=[lighten_color('black', 0.5)], alpha=1, zorder=1.1)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[3],
        linestyles=[(1, (4, 1))],
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("ICARUS (NuMI)")

    X1, Y1, Z1 = np.load(f"data/ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True)
    _, _, Z2 = np.load(f"data/ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = X1, Y1, Z1 + Z2
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles=[(1, (3, 2))],
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"Proto-DUNE")

    if plot_DUNEs:
        X, Y, Z = np.load(f"data/DUNE-ND_rates_{name}.npy", allow_pickle=True)
        c = ax.contour(
            X,
            Y**fa_power,
            Z,
            levels=[Nsig],
            colors="limegreen",
            linestyles=[(1, (1, 0))],
            linewidths=[1.75],
            alpha=1,
            zorder=2,
        )
        labels.append(c.legend_elements()[0][0])
        labelnames.append(r"DUNE ND")

        # X1,Y1,Z1 = np.load(f'data/2x2 protoDUNE-ND_rates_{name}.npy', allow_pickle=True)
        # _,_,Z2 = np.load(f'data/2x2 protoDUNE-ND absorber_rates_{name}.npy', allow_pickle=True)
        # X,Y,Z = X1, Y1, Z1+Z2
        # if smear:
        #     Z = scipy.ndimage.filters.gaussian_filter(Z, smear_stddev, mode="nearest", order=0, cval=0)
        # # c = ax.contourf(X, Y**fa_power, Z, levels=[Nsig, 1e100], colors=[lighten_color('green', 0.5)], alpha=1, zorder=-0.1)
        # c = ax.contour(X, Y**fa_power, Z, levels=[Nsig], colors=[lighten_color('green', 0.5)], linewidths=[1.75], alpha=1, zorder=2)
        # labels.append(c.legend_elements()[0][0])
        # labelnames.append(r'DUNE 2x2')

    # X1,Y1,Z1 = np.load(f'data/ArgoNeuT_rates_{name}.npy', allow_pickle=True)
    # _,_,Z2 = np.load(f'data/ArgoNeuT_absorber_rates_{name}.npy', allow_pickle=True)
    # X,Y,Z = X1, Y1, Z1+Z2
    # if smear:
    #     Z = scipy.ndimage.filters.gaussian_filter(Z, smear_stddev, mode="nearest", order=0, cval=0)
    # c = ax.contourf(X, Y**fa_power, Z, levels=[3, 1e100], colors=[lighten_color('green', 0.5)], alpha=1, zorder=-0.1)
    # _ = ax.contour(X, Y**fa_power, Z, levels=[3], colors='green', linewidths=[1.75], alpha=1, zorder=2)
    # labels.append(c.legend_elements()[0][0])
    # labelnames.append(r'ArgoNeuT (ours)')

    X, Y, Z = np.load(f"data/FASER_rates_{name}.npy", allow_pickle=True)
    # c = ax.contourf(X, Y**fa_power, Z, levels=[Nsig, 1e100], colors=[lighten_color='black', 0.95)], alpha=1, zorder=1.91)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(0, (2, 1))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("FASER")

    X, Y, Z = np.load(f"data/FASER2_rates_{name}.npy", allow_pickle=True)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(1, (5, 1))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"FASER-2")

    X, Y, Z = np.load(f"data/SHiP_rates_{name}.npy", allow_pickle=True)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles="-",
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    # x,y = c.collections[0].get_paths()[0].vertices[:,0],c.collections[0].get_paths()[0].vertices[:,1]
    # # x=np.append([1e-2], x)
    # # y=np.append([1e-2], y)
    # x,y=plot_closed_region((x,y), logx=True, logy=True)
    # c = ax.plot(x,y, edgecolor='black', facecolor='None', linestyle='-', linewidth=1.75, alpha=1, zorder=2)
    # labels.append(c[0])
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"SHiP")

    # ma, fa = np.genfromtxt(f"digitized/NuMI_{BP_NAME}.dat", delimiter=" ", unpack=True)
    # x, y = get_ordered_closed_region([ma, 1/fa], logx=True, logy=True)
    # labels.append(ax.fill(x,y, facecolor='None', edgecolor='red', linestyle='--', alpha=1, lw=1.5, label='Bertuzzo et al (2022)')[0])
    # labelnames.append(r'ArgoNeuT (Bertuzzo et al)')

    if legend:
        ax.legend(
            labels,
            labelnames,
            loc=loc,
            fontsize=9,
            ncol=ncol,
            frameon=False,
            framealpha=0.7,
            edgecolor="None",
            fancybox=True,
            handlelength=2.5,
            handletextpad=0.5,
            labelspacing=0.5,
            borderpad=0.5,
            columnspacing=0.75,
        )

    if not title:
        if c_lepton[0, 0] != c_lepton[0, 2]:
            title = r"{\bf LFV hierarchy} $\,\,\vert\,\,$"
            # ax.annotate(r'\noindent \bf LFV hierarchy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=11, ha='left', va='top')
            title += rf"$g_{{\ell \ell}} = {int(c_lepton[0,0])}$"
            title += rf"$\,\,\vert\,\, g_{{e\mu}} = g_{{e\tau}} = g_{{\mu\tau}} = {sci_notation(c_lepton[1,2], notex=True, precision=0)}$"
            # title+=rf'$\,\,\vert\,\, g_{{e\mu}} = \lambda^2 $'
            # title+=rf'$\,\,\vert\,\, \lambda = {sci_notation(c_lepton[1,2], notex=True, precision=0)}$'
            # title+=rf'$\,\,\vert\,\, g_{{(e,\mu)\tau}} = {sci_notation(c_lepton[1,2], notex=True, precision=1)}$'

        elif c_lepton[0, 0] == c_lepton[0, 2]:
            title = r"{\bf LFV anarchy} $\,\,\vert\,\,$"
            # ax.annotate(r'\noindent \bf LFV anarchy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=11, ha='left', va='top')
            title += rf"$g_{{\ell_1 \ell_2}} = {int(c_lepton[0,0])}$"
    ax.set_title(title, fontsize=11, pad=7.5)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylabel(rf"$f_a^{{{-fa_power}}}$ [GeV$^{{{-fa_power}}}$]")
    ax.set_xlabel(r"$m_a$ [GeV]")

    # ax.set_ylim(1e-10/2,1e-4)
    ax.set_ylim(ymin**fa_power, ymax**fa_power)
    ax.set_xlim(xmin, xmax)

    # ax.vlines(const.m_tau - const.m_e, ymin, ymax, color='black', linestyle='--', lw=0.5)
    # ax.annotate(r'$m_{\tau} - m_e$', (1.1*(const.m_tau - const.m_e), ymax/1.1), fontsize=9.5, ha='center', va='top', rotation=90)

    # ax.vlines(const.m_tau - const.m_mu, ymin, ymax, color='black', linestyle='--', lw=0.5)
    # ax.annotate(r'$m_{\tau} - m_\mu$', (0.92*(const.m_tau - const.m_mu), ymax/1.1), fontsize=9.5, ha='center', va='top', rotation=90)

    if vlines:
        ax.vlines(2 * const.m_mu, ymin, ymax, color="grey", linestyle="--", lw=0.5)
        ax.annotate(
            r"$2 m_\mu$",
            (0.92 * (2 * const.m_mu), ymax / 5),
            fontsize=9.5,
            ha="center",
            va="bottom",
            rotation=90,
            color="grey",
        )

        ax.vlines(
            const.m_mu + const.m_e, ymin, ymax, color="grey", linestyle="--", lw=0.5
        )
        ax.annotate(
            r"$m_\mu + m_e$",
            (0.92 * (const.m_mu + const.m_e), ymax / 5),
            fontsize=9.5,
            ha="center",
            va="bottom",
            rotation=90,
            color="grey",
        )

    # title+=rf'$\,\vert\, m_{{\psi}} = {mN}$'
    if grid:
        ax.grid(which="both")

    fig.savefig(
        f"plots/ALP_benchmark_{name}{name_modifier}.pdf", bbox_inches="tight", dpi=400
    )
    return fig, ax


def main_plot_LFV_bertuzzo_comparison(
    BP_NAME,
    c_lepton,
    c_NN,
    mN,
    fa_power=1,
    plot_DUNEs=True,
    figsize=(5, 6),
    ymax=1e-3,
    ymin=1e-10,
    xmin=1e-2,
    xmax=2,
    ncol=1,
    loc="lower left",
    yscale="log",
    xscale="log",
    legend=True,
    name_modifier="",
    vlines=True,
    title=None,
    smear_stddev=False,
    linewidth=0,
    annotate=False,
    plot_other=False,
    grid=False,
):

    fig, ax = std_fig(figsize=figsize)

    plot_other_limits(
        ax, c_lepton=c_lepton, c_NN=c_NN, mN=mN, linewidth=linewidth, annotate=annotate
    )

    # labels for legend
    labels = []
    labelnames = []
    name = BP_NAME

    Nsig = 3
    X1, Y1, Z1 = np.load(f"data/ArgoNeuT_rates_{name}.npy", allow_pickle=True)
    _, _, Z2 = np.load(f"data/ArgoNeuT_absorber_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = X1, Y1, Z1 + Z2
    c = ax.contourf(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color("green", 0.5)],
        alpha=1,
        zorder=-0.1,
    )
    _ = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors="green",
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"ArgoNeuT (ours)")

    if "Ra_5" in BP_NAME:
        ma, fa = np.genfromtxt(
            f"digitized/NuMI_LFV_Ra_5.dat", delimiter=" ", unpack=True
        )
    elif "Ra_1o3" in BP_NAME:
        ma, fa = np.genfromtxt(
            f"digitized/NuMI_LFV_Ra_1o3.dat", delimiter=" ", unpack=True
        )

    x, y = get_ordered_closed_region([ma, 1 / fa], logx=True, logy=True)
    labels.append(
        ax.fill(
            x,
            y,
            facecolor="None",
            edgecolor="red",
            linestyle="--",
            alpha=1,
            lw=1.5,
            label="Bertuzzo et al (2022)",
        )[0]
    )
    labelnames.append(r"ArgoNeuT (Bertuzzo et al)")

    if legend:
        ax.legend(
            labels,
            labelnames,
            loc=loc,
            fontsize=9,
            ncol=ncol,
            frameon=False,
            framealpha=0.7,
            edgecolor="None",
            fancybox=True,
            handlelength=2.5,
            handletextpad=0.5,
            labelspacing=0.5,
            borderpad=0.5,
            columnspacing=0.75,
        )

    if not title:
        if c_lepton[0, 0] != c_lepton[0, 2]:
            title = r"{\bf LFV hierarchy} $\,\,\vert\,\,$"
            # ax.annotate(r'\noindent \bf LFV hierarchy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=11, ha='left', va='top')
            title += rf"$g_{{\ell \ell}} = {int(c_lepton[0,0])}$"
            title += rf"$\,\,\vert\,\, g_{{e\mu}} = g_{{e\tau}} = g_{{\mu\tau}} = {sci_notation(c_lepton[1,2], notex=True, precision=0)}$"
            # title+=rf'$\,\,\vert\,\, g_{{e\mu}} = \lambda^2 $'
            # title+=rf'$\,\,\vert\,\, \lambda = {sci_notation(c_lepton[1,2], notex=True, precision=0)}$'
            # title+=rf'$\,\,\vert\,\, g_{{(e,\mu)\tau}} = {sci_notation(c_lepton[1,2], notex=True, precision=1)}$'

        elif c_lepton[0, 0] == c_lepton[0, 2]:
            title = r"{\bf LFV anarchy} $\,\,\vert\,\,$"
            # ax.annotate(r'\noindent \bf LFV anarchy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=11, ha='left', va='top')
            title += rf"$g_{{\ell_1 \ell_2}} = {int(c_lepton[0,0])}$"
    ax.set_title(title, fontsize=11, pad=7.5)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylabel(rf"$f_a^{{{-fa_power}}}$ [GeV$^{{{-fa_power}}}$]")
    ax.set_xlabel(r"$m_a$ [GeV]")

    # ax.set_ylim(1e-10/2,1e-4)
    ax.set_ylim(ymin**fa_power, ymax**fa_power)
    ax.set_xlim(xmin, xmax)

    # ax.vlines(const.m_tau - const.m_e, ymin, ymax, color='black', linestyle='--', lw=0.5)
    # ax.annotate(r'$m_{\tau} - m_e$', (1.1*(const.m_tau - const.m_e), ymax/1.1), fontsize=9.5, ha='center', va='top', rotation=90)

    # ax.vlines(const.m_tau - const.m_mu, ymin, ymax, color='black', linestyle='--', lw=0.5)
    # ax.annotate(r'$m_{\tau} - m_\mu$', (0.92*(const.m_tau - const.m_mu), ymax/1.1), fontsize=9.5, ha='center', va='top', rotation=90)

    if vlines:
        ax.vlines(2 * const.m_mu, ymin, ymax, color="grey", linestyle="--", lw=0.5)
        ax.annotate(
            r"$2 m_\mu$",
            (0.92 * (2 * const.m_mu), ymax / 5),
            fontsize=9.5,
            ha="center",
            va="bottom",
            rotation=90,
            color="grey",
        )

        ax.vlines(
            const.m_mu + const.m_e, ymin, ymax, color="grey", linestyle="--", lw=0.5
        )
        ax.annotate(
            r"$m_\mu + m_e$",
            (0.92 * (const.m_mu + const.m_e), ymax / 5),
            fontsize=9.5,
            ha="center",
            va="bottom",
            rotation=90,
            color="grey",
        )

    # title+=rf'$\,\vert\, m_{{\psi}} = {mN}$'
    if grid:
        ax.grid(which="both")

    fig.savefig(
        f"plots/ALP_benchmark_{name}{name_modifier}.pdf", bbox_inches="tight", dpi=400
    )
    return fig, ax


def main_plot_LFC(
    BP_NAME,
    c_lepton,
    fa_power=1,
    plot_DUNEs=True,
    figsize=(5, 5),
    ymax=1e-3,
    ymin=1e-10,
    xmin=1e-2,
    xmax=2,
    ncol=1,
    loc="upper right",
    yscale="log",
    xscale="log",
    legend=True,
    name_modifier="",
    vlines=True,
    linewidth=0.5,
    plot_other_limits=True,
):

    fig, ax = std_fig(figsize=figsize)
    if plot_other_limits:
        plot_other_limits_LFC(ax, linewidth=linewidth, c_lepton=c_lepton)

    # labels for legend
    labels = []
    labelnames = []
    name = BP_NAME

    Nsig = 2.3
    X, Y, Z = np.load(f"data/CHARM_rates_{name}.npy", allow_pickle=True)
    c = ax.contourf(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[0], 0.5)],
        alpha=1,
        zorder=1.2,
    )
    _ = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("CHARM")

    X, Y, Z = np.load(f"data/BEBC_rates_{name}.npy", allow_pickle=True)
    c = ax.contourf(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[1], 0.5)],
        alpha=1,
        zorder=1.2,
    )
    _ = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[1],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("BEBC")

    X1, Y1, Z1 = np.load(f"data/ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True)
    _, _, Z2 = np.load(f"data/ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = X1, Y1, Z1 + Z2
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles=[(1, (3, 2))],
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"ProtoDUNE")

    if plot_DUNEs:
        X, Y, Z = np.load(f"data/DUNE-ND_rates_{name}.npy", allow_pickle=True)
        c = ax.contour(
            X,
            Y**fa_power,
            Z,
            levels=[Nsig],
            colors="limegreen",
            linestyles=[(1, (1, 0))],
            linewidths=[1.75],
            alpha=1,
            zorder=2,
        )
        labels.append(c.legend_elements()[0][0])
        labelnames.append(r"DUNE ND")

        # X,Y,Z = np.load(f'data/2x2 protoDUNE-ND_rates_{name}.npy', allow_pickle=True)
        # if smear:
        #     Z = scipy.ndimage.filters.gaussian_filter(Z, smear_stddev, mode="nearest", order=0, cval=0)
        # c = ax.contour(X, Y**fa_power, Z, levels=[Nsig], colors='red', linestyles=[(1,(1,1))], linewidths=[1.75], alpha=1, zorder=2)
        # labels.append(c.legend_elements()[0][0])
        # labelnames.append(r'2x2 protoDUNE ND')

    # X,Y,Z = np.load(f'data/FASER_rates_{name}.npy', allow_pickle=True)
    # if smear:
    #     Z = scipy.ndimage.filters.gaussian_filter(Z, 2*smear_stddev, mode="nearest", order=0, cval=0)
    # # c = ax.contourf(X, Y**fa_power, Z, levels=[Nsig, 1e100], colors=[lighten_color='black', 0.95)], alpha=1, zorder=1.91)
    # c = ax.contour(X, Y**fa_power, Z, levels=[Nsig], colors='black', linestyles=[(0,(2,2))], linewidths=[1.5], alpha=1, zorder=2)
    # labels.append(c.legend_elements()[0][0])
    # labelnames.append('FASER')

    X, Y, Z = np.load(f"data/FASER2_rates_{name}.npy", allow_pickle=True)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(1, (5, 1))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"FASER-2")

    X, Y, Z = np.load(f"data/SHiP_rates_{name}.npy", allow_pickle=True)
    c = ax.contour(
        X,
        Y**fa_power,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles="-",
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"SHiP")

    if legend:
        ax.legend(
            labels,
            labelnames,
            loc=loc,
            fontsize=9,
            ncol=ncol,
            frameon=False,
            framealpha=0.8,
            edgecolor="black",
            fancybox=False,
            handlelength=2.5,
            handletextpad=0.5,
            labelspacing=0.5,
            borderpad=0.5,
        )

    if c_lepton[0, 0] == c_lepton[2, 2] and c_lepton[1, 1] == c_lepton[2, 2]:
        title = r"{\bf LFC flavor universal} $\,\,\vert\,\,$"
        title += rf"$g_{{\ell \ell}} = {int(c_lepton[0,0])}$"
    elif c_lepton[1, 1] == 0 and c_lepton[0, 0] > 0:
        title = r"{\bf LFC $\mu$-phobic} $\,\,\vert\,\,$"
        title += rf"$g_{{e e}} = g_{{\tau \tau}} = {int(c_lepton[0,0])}$ $\,\,\vert\,\,$ $g_{{\mu \mu}} = {int(c_lepton[1,1])}$"
    elif c_lepton[1, 1] == 0 and c_lepton[0, 0] == 0 and c_lepton[2, 2] > 0:
        title = r"{\bf LFC $\tau$-philic} $\,\,\vert\,\,$"
        title += rf"$g_{{e e}} = g_{{\mu \mu}} = {int(c_lepton[0,0])}$ $\,\,\vert\,\,$ $g_{{\tau \tau}} = {int(c_lepton[2,2])}$"

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylabel(rf"$f_a^{{{-fa_power}}}$ [GeV$^{{{-fa_power}}}$]")
    ax.set_xlabel(r"$m_a$ [GeV]")

    ax.set_ylim(ymin**fa_power, ymax**fa_power)
    ax.set_xlim(xmin, xmax)

    if vlines:
        ax.vlines(2 * const.m_mu, ymin, ymax, color="grey", linestyle="--", lw=0.5)
        ax.annotate(
            r"$2 m_\mu$",
            (0.92 * (2 * const.m_mu), ymin * 1.1),
            fontsize=9.5,
            ha="center",
            va="bottom",
            rotation=90,
            color="grey",
        )

        ax.vlines(
            const.m_mu + const.m_e, ymin, ymax, color="grey", linestyle="--", lw=0.5
        )
        ax.annotate(
            r"$m_\mu + m_e$",
            (0.92 * (const.m_mu + const.m_e), ymin * 1.1),
            fontsize=9.5,
            ha="center",
            va="bottom",
            rotation=90,
            color="grey",
        )

    ax.set_title(title, fontsize=11, pad=7.5)

    fig.savefig(
        f"plots/ALP_benchmark_{name}{name_modifier}.pdf", bbox_inches="tight", dpi=400
    )
    return fig, ax


# def plot_other_limits_Bvis(
#     ax, ma_fixed, Bvis_range=[1e-12, 1], c_lepton=None, c_NN=0, mN=0, linewidth=0.25
# ):

#     # BaBar Limit
#     ma, qsi = np.genfromtxt("digitized/BABAR_leptophilic.dat", unpack=True)
#     ax.fill_between(
#         ma,
#         (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
#         qsi / qsi,
#         color="silver",
#         linestyle="-",
#     )
#     ax.plot(
#         ma,
#         (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
#         color="black",
#         linestyle="-",
#         lw=linewidth,
#     )
#     # BELLE Limit
#     ma, qsi = np.genfromtxt("digitized/Belle_leptophilic.dat", unpack=True)
#     ax.fill_between(
#         ma,
#         (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
#         qsi / qsi,
#         color="silver",
#         linestyle="-",
#     )
#     ax.plot(
#         ma,
#         (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
#         color="black",
#         linestyle="-",
#         lw=linewidth,
#     )

#     Bvis = np.geomspace(Bvis_range[0], Bvis_range[1], 1000)
#     inv_fa = np.geomspace(1e-9, 1e-2, 1000, endpoint=True)
#     BVIS, INV_FA = np.meshgrid(Bvis, inv_fa)

#     alps = ALP(ma_fixed, 1 / INV_FA, c_lepton=c_lepton, c_NN=c_NN, mN=mN, Bvis=BVIS)
#     ################################################################
#     # Belle-II (Tau -> mu alp)
#     ma_limit, B_limit_90CL = np.genfromtxt(
#         "digitized/BelleII_tau_to_mu_a.dat", unpack=True
#     )
#     B_limit_90CL_interp = np.interp(
#         ma_fixed, ma_limit, B_limit_90CL, left=1e100, right=1e100
#     )

#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

#     BR_tau_mu_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_mu()
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["silver"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle-II (Tau -> e alp)
#     ma_limit, B_limit_90CL = np.genfromtxt(
#         "digitized/BelleII_tau_to_e_a.dat", unpack=True
#     )
#     B_limit_90CL_interp = np.interp(
#         ma_fixed, ma_limit, B_limit_90CL, left=1e100, right=1e100
#     )

#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

#     BR_tau_e_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_e()
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_e_a,
#         levels=[1, 1e100],
#         colors=["silver"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_e_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle (Tau -> mu alp)
#     ma_limit, B_limit_90CL = np.genfromtxt(
#         "digitized/Belle_tau_to_mu_a.dat", unpack=True
#     )
#     B_limit_90CL_interp = np.interp(
#         ma_fixed, ma_limit, B_limit_90CL, left=1e100, right=1e100
#     )

#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

#     BR_tau_mu_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_mu()
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["silver"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle (Tau -> e alp)
#     ma_limit, B_limit_90CL = np.genfromtxt(
#         "digitized/Belle_tau_to_e_a.dat", unpack=True
#     )
#     B_limit_90CL_interp = np.interp(
#         ma_fixed, ma_limit, B_limit_90CL, left=1e100, right=1e100
#     )

#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

#     BR_tau_e_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_e()
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_e_a,
#         levels=[1, 1e100],
#         colors=["silver"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_e_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Total tau lifetime
#     BR_tau_ell_a = alps.BR_tau_to_a_e() + alps.BR_tau_to_a_mu()
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_ell_a,
#         levels=[1e-2, 1e100],
#         colors=[lighten_color(inv_color, 0.5)],
#         alpha=0.75,
#         zorder=0.2,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_ell_a,
#         levels=[1e-2],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle (Tau- -> e- (alp -> mu+ mu-)) + (Tau- -> mu- (alp -> mu+ e-))
#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
#     B_limit_90CL = 2.7e-8
#     BR_tau_mu_a = (
#         1
#         / B_limit_90CL
#         * P_decay
#         * (
#             alps.BR_tau_to_a_e() * alps.BR_a_to_mm
#             + alps.BR_tau_to_a_mu() * alps.BR_a_to_me
#         )
#     )
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["grey"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle (Tau- -> mu- (alp -> mu+ mu-))
#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
#     B_limit_90CL = 2.1e-8
#     BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_mm
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["grey"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle (Tau- -> mu- (alp -> mu+ e-))
#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
#     B_limit_90CL = 2.7e-8
#     BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["grey"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     # Belle (Tau- -> mu- (alp -> mu- e+))
#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
#     B_limit_90CL = 1.7e-8
#     BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_em
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["grey"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
#     )

#     ################################################################
#     # Belle (Tau- -> e- (alp -> mu+ e-))
#     p_alp_avg = 10.58 / 4 * (1 + MA**2/const.m_tau**2 - mlepton**2/const.m_tau**2)
#     P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
#     B_limit_90CL = 1.5e-8
#     BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
#     _ = ax.contourf(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1, 1e100],
#         colors=["grey"],
#         alpha=1,
#         zorder=0,
#     )
#     _ = ax.contour(
#         BVIS,
#         INV_FA,
#         BR_tau_mu_a,
#         levels=[1],
#         colors=["black"],
#         linestyles="-",
#         linewidths=[linewidth],
#         alpha=1,
#         zorder=2,
# )


def plot_other_limits(
    ax,
    c_lepton=None,
    c_NN=0,
    mN=0,
    linewidth=0.25,
    vis_color=lighten_color("lightgrey", 0.3),
    inv_color="lightgrey",
    annotate=False,
    edgecolor="dimgrey",
):

    # Muon Limits
    x, y = np.genfromtxt("digitized/Jodidio_et_al_A.dat", unpack=True)
    y *= 2
    x *= 1e-9
    ax.fill_between(
        x,
        (1 / y) / c_lepton[0, 1],
        y / y,
        facecolor=inv_color,
        linestyle="-",
        zorder=0.1,
        edgecolor="None",
    )
    ax.plot(
        x,
        (1 / y) / c_lepton[0, 1],
        color="dimgrey",
        linestyle="-",
        lw=0.2,
        zorder=0.2,
    )

    # x, y = np.genfromtxt("digitized/TWIST_mue_A.dat", unpack=True)
    # y *= 2
    # x *= 1e-9
    # y = y[np.argsort(x)]
    # x = x[np.argsort(x)]
    # ax.fill_between(
    #     x,
    #     (1 / y) / c_lepton[0, 1],
    #     y / y,
    #     facecolor=inv_color,
    #     edgecolor="None",
    #     linestyle="-",
    #     zorder=0.1,
    # )
    # ax.plot(
    #     x,
    #     (1 / y) / c_lepton[0, 1],
    #     color=edgecolor,
    #     linestyle="-",
    #     lw=0.2,
    #     zorder=0.2,
    # )

    # SN1987A
    x, y = np.genfromtxt("digitized/Supernova_mumu.dat", unpack=True)
    x *= 1e-9
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    ax.fill_between(
        x,
        (1 / y) / c_lepton[1, 1],
        y / y,
        facecolor=lighten_color(inv_color, 0.5),
        edgecolor="None",
        linestyle="-",
        zorder=0.1,
    )
    ax.plot(
        x,
        (1 / y) / c_lepton[1, 1],
        color=lighten_color(edgecolor, 0.5),
        linestyle=(1, (6, 2)),
        lw=0.2,
        zorder=0.2,
    )

    # SN1987A ee
    x, y = np.genfromtxt("digitized/Supernova_ee.dat", unpack=True)
    x *= 1e-9
    x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
    ax.fill(
        x,
        (1 / y) / c_lepton[0, 0],
        facecolor=lighten_color(inv_color, 0.5),
        edgecolor="None",
        linestyle="-",
        zorder=0.1,
    )
    ax.plot(
        x,
        (1 / y) / c_lepton[0, 0],
        color=lighten_color(edgecolor, 0.5),
        linestyle=(1, (6, 2)),
        lw=0.2,
        zorder=0.2,
    )

    # SN1987A emu
    for v in ["v1", "v2"]:
        x, y = np.genfromtxt(f"digitized/Supernova_emu_{v}.dat", unpack=True)
        x *= 1e-3
        y *= 2 / (const.m_e + const.m_mu)
        x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
        ax.fill(
            x,
            y / c_lepton[0, 1],
            facecolor=lighten_color(inv_color, 0.5),
            edgecolor="None",
            linestyle="-",
            zorder=0.1,
            alpha=0.75,
        )
        ax.plot(
            x,
            y / c_lepton[0, 1],
            color=lighten_color(edgecolor, 0.5),
            linestyle=(1, (6, 2)),
            lw=0.5,
            zorder=0.2,
        )

    # PIENU bounds
    for b in [
        "digitized/PIENU_mu_to_e_X.dat",
        "digitized/Derenzo_et_al_mu_to_e_X.dat",
        "digitized/Bilger_et_al_mu_to_e_X.dat",
    ]:
        mX, BR = np.genfromtxt(b, unpack=True)
        mX *= 1e-3
        alp = ALP(mX, 1.0, c_lepton=c_lepton, c_NN=c_NN, mN=mN)
        inv_fa = np.sqrt(BR / alp.BR_li_to_lj_a(1, 0))
        ax.fill_between(
            mX,
            inv_fa,
            inv_fa / inv_fa,
            edgecolor="None",
            facecolor=inv_color,
            linestyle="-",
            zorder=0.02,
        )
        ax.plot(
            mX,
            inv_fa,
            color="dimgrey",
            linestyle="-",
            lw=0.25,
            zorder=1,
        )

    pe, BR = np.genfromtxt("digitized/TWIST_mu_to_e_X.dat", unpack=True)
    pe *= 1e-3
    mX = np.sqrt(
        const.m_e**2 + const.m_mu**2 - 2 * const.m_mu * np.sqrt(pe**2 + const.m_e**2)
    )
    BR = np.append(BR, BR[np.argmin(mX)])
    mX = np.append(mX, 0)

    alp = ALP(mX, 1.0, c_lepton=c_lepton, c_NN=c_NN, mN=mN)
    inv_fa = np.sqrt(BR / alp.BR_li_to_lj_a(1, 0))
    ax.fill_between(
        mX,
        inv_fa,
        inv_fa / inv_fa,
        edgecolor="None",
        facecolor=inv_color,
        linestyle="-",
        zorder=0.01,
    )
    ax.plot(
        mX,
        inv_fa,
        color="dimgrey",
        linestyle="-",
        lw=0.25,
        zorder=0.2,
    )

    # ###########################################################
    # ma, y = np.genfromtxt("digitized/E137_Araki_et_al_LFV.dat", unpack=True)
    # inv_fa = y / const.m_e
    # x, y = get_ordered_closed_region([ma, inv_fa], logx=True, logy=True)
    # ax.plot(
    #     x,
    #     y,
    #     color="black",
    #     linestyle="-",
    #     zorder=3,
    # )

    ###########################################################
    ma = np.geomspace(1e-3, 3, 100)
    inv_fa = np.geomspace(1e-9, 1e-2, 100, endpoint=True)
    MA, INV_FA = np.meshgrid(ma, inv_fa)

    alps = ALP(MA, 1 / INV_FA, c_lepton=c_lepton, c_NN=c_NN, mN=mN, Bvis=1)

    ################################################################
    # LFV MEG-II (mu -> e gamma)
    Z = alps.BR_li_to_lj_gamma(1, 0)
    ax.contourf(
        MA, INV_FA, Z, levels=[1.5e-13, 1e10], colors=[inv_color], alpha=1, zorder=0.2
    )
    ax.contour(
        MA,
        INV_FA,
        Z,
        levels=[1.5e-13],
        alpha=1,
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        zorder=2,
    )
    #
    ################################################################
    # LFV BaBAr (tau -> ell gamma)
    Z = alps.BR_li_to_lj_gamma(2, 0)
    ax.contourf(
        MA, INV_FA, Z, levels=[3.3e-8, 1e10], colors=[inv_color], alpha=1, zorder=0.2
    )
    ax.contour(
        MA,
        INV_FA,
        Z,
        levels=[3.3e-8],
        alpha=1,
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        zorder=2,
    )
    #
    Z = alps.BR_li_to_lj_gamma(2, 1)
    ax.contourf(
        MA, INV_FA, Z, levels=[4.2e-8, 1e10], colors=[inv_color], alpha=1, zorder=0.2
    )
    ax.contour(
        MA,
        INV_FA,
        Z,
        levels=[4.2e-8],
        alpha=1,
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        zorder=2,
    )
    #

    ###########################################################
    ma = np.geomspace(1e-3, 3, 1000)
    inv_fa = np.geomspace(1e-9, 1e-3, 1000, endpoint=True)
    MA, INV_FA = np.meshgrid(ma, inv_fa)

    alps = ALP(MA, 1 / INV_FA, c_lepton=c_lepton, c_NN=c_NN, mN=mN, Bvis=1)

    # ################################################################
    # # Belle-II (Tau -> mu alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/BelleII_tau_to_mu_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(MA, ma_limit, B_limit_90CL, left=1e100, right=1e100)

    mlepton = const.m_mu
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

    BR_tau_mu_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_mu()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[inv_color],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    # ################################################################
    # # Belle-II (Tau -> e alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/BelleII_tau_to_e_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(MA, ma_limit, B_limit_90CL, left=1e100, right=1e100)

    mlepton = const.m_e
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

    BR_tau_e_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_e()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_e_a,
        levels=[1, 1e100],
        colors=[inv_color],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_e_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau -> mu alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/Belle_tau_to_mu_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(
        MA, ma_limit, B_limit_90CL, left=np.inf, right=np.inf
    )

    mlepton = const.m_mu
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 300, np.inf)  # 3 m travel

    BR_tau_mu_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_mu()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, np.inf],
        colors=[inv_color],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau -> e alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/Belle_tau_to_e_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(
        MA, ma_limit, B_limit_90CL, left=np.inf, right=np.inf
    )
    mlepton = const.m_e
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 300, np.inf)  # 3 m travel

    BR_tau_e_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_e()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_e_a,
        levels=[1, np.inf],
        colors=[inv_color],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_e_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Total tau lifetime
    BR_tau_ell_a = alps.BR_tau_to_a_e() + alps.BR_tau_to_a_mu()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_ell_a,
        levels=[1e-3, 1e100],
        colors=[inv_color],
        alpha=1,
        zorder=0.19,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_ell_a,
        levels=[1e-3],
        colors=["black"],
        linestyles="--",
        linewidths=[4 * linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> e- (alp -> mu+ mu-)) + (Tau- -> mu- (alp -> mu+ e-))
    mlepton = const.m_e
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.7e-8
    BR_tau_mu_a = (
        1
        / B_limit_90CL
        * P_decay
        * (
            alps.BR_tau_to_a_e() * alps.BR_a_to_mm
            + alps.BR_tau_to_a_mu() * alps.BR_a_to_me
        )
    )
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> mu- (alp -> mu+ mu-))
    mlepton = const.m_mu
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.9e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_mm
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> mu- (alp -> mu+ e-))
    mlepton = const.m_mu
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.7e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    # Belle (Tau- -> mu- (alp -> mu- e+))
    mlepton = const.m_mu
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.7e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_em
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> e- (alp -> mu+ e-))
    mlepton = const.m_e
    p_alp_avg = 10.58 / 4 * (1 + MA**2 / const.m_tau**2 - mlepton**2 / const.m_tau**2)
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.5e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
    _ = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    _ = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )
    if annotate:
        ax.annotate(
            r"Supernova $g_{\mu \mu}$",
            xy=(1.05e-2, 2.4e-8 / c_lepton[1, 1]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"Supernova $g_{ee}$",
            xy=(1.05e-2, 8.5e-8 / c_lepton[0, 0]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"Supernova $g_{e\mu}$",
            xy=(1.05e-2, 0.65e-8 / c_lepton[0, 1]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        if 8e-10 / c_lepton[0, 1] < 5e-4:
            # if c_lepton[0, 1] == 1:
            #     ax.annotate(
            #         r"$\mu\to e a_{\rm inv}$",
            #         xy=(1.05e-2, 1.0e-9),
            #         xycoords="data",
            #         fontsize=9,
            #         horizontalalignment="left",
            #         verticalalignment="center",
            #         rotation=0,
            #         zorder=3,
            #         color="dimgrey",
            #     )
            # else:
            ax.annotate(
                r"$\mu\to e a_{\rm inv}$",
                xy=(9e-2, 1.8e-9 / c_lepton[0, 1]),
                xycoords="data",
                fontsize=9,
                horizontalalignment="left",
                verticalalignment="center",
                rotation=90,
                zorder=3,
                color="dimgrey",
            )

        ax.annotate(
            r"$\tau \to \ell a_{\rm inv}$",
            xy=(1.05e-2, 3e-7 / c_lepton[0, 2]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"$\tau \to \mu \gamma$",
            xy=(1.05e-2, 1.3e-4 / np.sqrt(c_lepton[2, 1])),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"$\tau \to e \gamma$",
            xy=(1.05e-2, 0.6e-4 / np.sqrt(c_lepton[2, 0])),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"$\mu \to e \gamma$",
            xy=(1.05e-2, 2.2e-5 / np.sqrt(c_lepton[1, 0])),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"$\tau$ lifetime",
            xy=(0.65, 2 * 4.5e-7 / (c_lepton[1, 2] + c_lepton[0, 2])),
            xycoords="data",
            rotation=20,
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color="black",
        )
        ax.annotate(
            r"$\tau \to \ell a_{\rm vis}$",
            xy=(0.8, 1.2e-7 / np.sqrt(c_lepton[1, 2])),
            xycoords="data",
            rotation=0,  # 20,
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color="black",
        )


def plot_other_limits_LFC(
    ax, c_lepton, alp=None, linewidth=1, nocolor="lightgrey", edgecolor="dimgrey"
):

    ###########################################################
    # ma, y = np.genfromtxt("digitized/E137_Araki_et_al_LFC.dat", unpack=True)
    # inv_fa = y / const.m_e
    # ma[ma > 2 * const.m_mu] = 2*const.m_mu
    # x, y = get_ordered_closed_region([ma, inv_fa], logx=True, logy=True)
    # ax.fill(
    #     x,
    #     y,
    #     facecolor=nocolor,
    #     edgecolor="red",
    #     linestyle="-",
    #     zorder=1.1,
    #     linewidth=1,
    #     # label='E137 (Araki et al. 2021)',
    # )

    if c_lepton[0, 0] > 0:

        # if c_lepton[0, 0] > 0:
        #     ###########################################################
        #     ma = np.geomspace(1e-3, 10, 1000)
        #     fa = np.geomspace(0.1, 1e6, 1000)
        #     M, F = np.meshgrid(ma, fa)

        #     alp = ALP(M, F, c_lepton=c_lepton)
        #     Z = alp.delta_a_mag_mom(l_i=0)
        #     ax.contourf(
        #         M,
        #         1 / F,
        #         Z,
        #         levels=[
        #             DELTA_a_electron_LKB[0] - 2 * DELTA_a_electron_LKB[1],
        #             DELTA_a_electron_LKB[0] + 2 * DELTA_a_electron_LKB[1],
        #         ],
        #         colors=["lightgrey"],
        #         linewidths=[0],
        #     )

        #     ax.contour(
        #         M,
        #         1 / F,
        #         Z,
        #         levels=[
        #             DELTA_a_electron_LKB[0] - 2 * DELTA_a_electron_LKB[1],
        #             DELTA_a_electron_LKB[0] + 2 * DELTA_a_electron_LKB[1],
        #         ],
        #         colors=[edgecolor],
        #         linestyles=["-"],
        #         linewidths=linewidth * 2,
        #         zorder=2,
        #     )
        #     ax.annotate(
        #         r"$(g-2)_e$",
        #         xy=(0.4, 5e-3),
        #         fontsize=10,
        #         color="grey",
        #         ha="left",
        #         va="bottom",
        #         rotation=18,
        #     )

        if c_lepton[1, 1] > 0:
            ###########################################################
            ma = np.geomspace(1e-3, 10, 1000)
            fa = np.geomspace(0.1, 1e6, 1000)
            M, F = np.meshgrid(ma, fa)

            alp = ALP(M, F, c_lepton=c_lepton)
            Z = alp.delta_a_mag_mom(l_i=1)
            ax.contourf(
                M,
                1 / F,
                Z,
                levels=[
                    -np.inf,
                    DELTA_COMBINED_2023_SM_2025[0] - 2 * DELTA_COMBINED_2023_SM_2025[1],
                ],
                colors=["lightgrey"],
                linewidths=[0],
            )

            ax.contour(
                M,
                1 / F,
                Z,
                levels=[
                    DELTA_COMBINED_2023_SM_2025[0] - 2 * DELTA_COMBINED_2023_SM_2025[1],
                ],
                colors=[edgecolor],
                linestyles=["-"],
                linewidths=linewidth * 2,
                zorder=2,
            )
            ax.annotate(
                r"$(g-2)_\mu$",
                xy=(0.4, 5e-3),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=18,
            )

        if c_lepton[0, 0] == c_lepton[1, 1] and c_lepton[1, 1] == c_lepton[2, 2]:
            ###########################################################
            x, y = np.genfromtxt("digitized/NA64mu_LFC.dat", unpack=True)
            x = x * 1e-3
            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=0.4,
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=1.01,
                linewidth=2 * linewidth,
            )
            ax.annotate(
                r"NA64$\mu$ (inv)",
                xy=(0.04, 0.004),
                fontsize=10,
                rotation=-39,
                color="grey",
                ha="left",
                va="bottom",
            )

            ###########################################################
            x, y = np.genfromtxt("digitized/E137_LFC.dat", unpack=True)
            x = x * 1e-3
            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=0.4,
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=2,
                linewidth=2 * linewidth,
            )
            ax.annotate(
                r"E137 (Eberhart et al)",
                xy=(4e-3, 0.3e-4),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=0,
            )
            ###########################################################
            x, y = np.genfromtxt("digitized/NA64vis_LFC.dat", unpack=True)
            x = x * 1e-3
            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=0.4,
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=2,
                linewidth=2 * linewidth,
            )
            ax.annotate(
                r"NA64e (vis)",
                xy=(7e-3, 1.8e-1),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=-44,
            )

        ###########################################################
        x, y = np.genfromtxt("digitized/Orsay_Liu_et_al.dat", unpack=True)
        x = x * 1e-3
        y = const.eQED * y / const.m_e
        x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
        ax.fill(
            x,
            y,
            facecolor="lightgrey",
            edgecolor="None",
            linestyle="-",
            zorder=0.4,
            # label='E137 (Araki et al. 2021)',
        )
        ax.fill(
            x,
            y,
            facecolor="None",
            edgecolor=edgecolor,
            linestyle="-",
            zorder=1.01,
            linewidth=2 * linewidth,
            # label='E137 (Araki et al. 2021)',
        )
        ax.annotate(
            r"Orsay",
            xy=(1.6e-2, 1e-2),
            fontsize=10,
            color="grey",
            ha="left",
            va="bottom",
            rotation=-36,
        )
        # ###########################################################
        # x, y = np.genfromtxt("digitized/NA64_pseudoscalar.dat", unpack=True)
        # x = x * 1e-3
        # y = const.eQED * y / const.m_e
        # x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
        # ax.fill(
        #     x,
        #     y,
        #     facecolor="lightgrey",
        #     edgecolor="None",
        #     linestyle="-",
        #     zorder=0.4,
        # )
        # ax.fill(
        #     x,
        #     y,
        #     facecolor="None",
        #     edgecolor=edgecolor,
        #     linestyle="-",
        #     zorder=1.01,
        #     linewidth=2 * linewidth,
        # )

        ###########################################################
        x, y = np.genfromtxt("digitized/E141_Liu_et_al.dat", unpack=True)
        x = x * 1e-3
        y = const.eQED * y / const.m_e
        x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
        ax.fill(
            x,
            y,
            facecolor="lightgrey",
            edgecolor="None",
            linestyle="-",
            zorder=0.4,
            # label='E137 (Araki et al. 2021)',
        )
        ax.fill(
            x,
            y,
            facecolor="None",
            edgecolor=edgecolor,
            linestyle="-",
            zorder=1.01,
            linewidth=2 * linewidth,
            # label='E137 (Araki et al. 2021)',
        )

        # Create a new figure with a specific size
        norm = 1 / 2 / const.m_e  # gee

        ma, ge = np.genfromtxt("digitized/SN1987A_cooling_cool.dat", unpack=True)
        ma = ma * 1e-3
        inv_fa = ge * norm
        x, y = get_ordered_closed_region([ma, inv_fa], logx=True, logy=True)
        ax.fill(
            x,
            y,
            facecolor=nocolor,
            edgecolor=edgecolor,
            alpha=1,
            zorder=1.1,
            linewidth=linewidth,
        )

        # ma, ge = np.genfromtxt('digitized/SN1987A_B=1_cool.dat', unpack=True)
        # ma = ma * 1e-3
        # inv_fa = ge *norm
        # x,y=get_ordered_closed_region([ma, inv_fa], logx=True, logy=True)

        # ax.plot(x, y, ls=(1,(1,1)), color='black', alpha=1, zorder=2, linewidth=linewidth*2)

        ma, ge = np.genfromtxt("digitized/SN1987A_B=0.1_cool.dat", unpack=True)
        ma = ma * 1e-3
        inv_fa = ge * norm
        x, y = get_ordered_closed_region([ma, inv_fa], logx=True, logy=True)
        plt.plot(x, y, color="silver", zorder=1.1, linewidth=1)
        ax.fill(
            x,
            y,
            facecolor=lighten_color(nocolor, 0.5),
            edgecolor="black",
            alpha=1,
            zorder=-1.1,
            linewidth=linewidth,
        )

        # BaBar Limit
        ma, qsi = np.genfromtxt("digitized/BABAR_leptophilic.dat", unpack=True)
        ax.fill_between(
            ma,
            qsi / const.vev_EW,
            qsi / qsi * 100,
            color=nocolor,
            edgecolor="None",
            linestyle="-",
        )
        ax.plot(
            ma,
            qsi / const.vev_EW,
            color=edgecolor,
            linestyle="-",
            lw=linewidth,
        )

        # Belle Limit
        ma, qsi = np.genfromtxt("digitized/Belle_leptophilic.dat", unpack=True)
        ax.fill_between(
            ma,
            qsi / const.vev_EW,
            qsi / qsi,
            color="silver",
            edgecolor="None",
            linestyle="-",
        )
        ax.plot(
            ma,
            qsi / const.vev_EW,
            color=edgecolor,
            linestyle="-",
            lw=linewidth,
        )
        ###########################################################
        if c_lepton[0, 0] > 0 and c_lepton[1, 1] == 0:

            ###########################################################
            x, y = np.genfromtxt("digitized/NA64vis_LFC.dat", unpack=True)
            x = x * 1e-3
            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=0.4,
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=2,
                linewidth=2 * linewidth,
            )
            ax.annotate(
                r"NA64e (vis)",
                xy=(7e-3, 1.6e-1),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=-44,
            )

            x, y = np.genfromtxt("digitized/E137_Araki_et_al_LFC.dat", unpack=True)
            y = y / const.m_e
            # y = np.append(y, y[np.argmin(x*y)])
            # x = np.append(x, 1e-3)
            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=0.4,
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=1.01,
                linewidth=2 * linewidth,
            )

            x, y = np.genfromtxt("digitized/E137_Liu_et_al_LFC.dat", unpack=True)
            x = x * 1e-3
            y = const.eQED * y / const.m_e
            y = y[x > 1e-3]
            x = x[x > 1e-3]

            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=0.4,
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=1.01,
                linewidth=2 * linewidth,
            )

        ###########################################################
        # Dror et al. limits

        # Load the data from the files
        data1 = np.loadtxt("digitized/CHARM.dat")
        data1[:, 0] *= 1e-3
        data1[:, 1] *= 2
        data2 = np.loadtxt("digitized/electron_gminus2.dat")
        data2[:, 0] *= 1e-3
        data2[:, 1] *= 2
        data3 = np.loadtxt("digitized/pion_decay_weak_preserving.dat")
        data3[:, 0] *= 1e-3
        data3[:, 1] *= 2

        x, y = get_ordered_closed_region(data1.T, logx=True, logy=True)
        ax.fill(x, norm * y, facecolor=nocolor, edgecolor="None", alpha=1, zorder=1)
        if linewidth:
            ax.fill(
                x,
                norm * y,
                facecolor="None",
                edgecolor=edgecolor,
                ls=(1, (3, 1)),
                lw=linewidth * 2,
                zorder=1.1,
            )

        ax.fill_between(
            data2[:, 0],
            norm * data2[:, 1],
            1e5 * np.ones(len(data2[:, 1])),
            facecolor=nocolor,
            edgecolor="None",
            alpha=1,
            zorder=1,
        )
        ax.fill_between(
            data3[:, 0],
            norm * data3[:, 1],
            1e5 * np.ones(len(data3[:, 1])),
            facecolor=nocolor,
            edgecolor="None",
            alpha=1,
            zorder=1,
        )
        ax.fill_between(
            data3[:, 0],
            norm * data3[:, 1],
            1e5 * np.ones(len(data3[:, 1])),
            edgecolor=edgecolor,
            facecolor="None",
            lw=linewidth,
            zorder=1.1,
        )

        # Add text labels to the gray regions
        if linewidth:
            ax.annotate(
                r"SN cooling ($g_{ee}$)",
                xy=(2.4e-3, 1.2e-5 / 2),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
            )
            ax.annotate(
                r"Low-energy SN ($g_{ee}$)",
                xy=(0.18, 1.2e-5 / 2),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
            )
            ax.annotate(
                r"CHARM (Dror et al)",
                xy=(0.002, norm * 2.2e-7),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=-26,
            )

            if c_lepton[0, 0] > 0 and c_lepton[1, 1] == 0:
                ax.annotate(
                    r"E137 (Araki et al)",
                    xy=(2e-2, norm * 2.5e-8),
                    fontsize=10,
                    color="grey",
                    ha="left",
                    va="bottom",
                    rotation=0,
                )
                ax.annotate(
                    r"E137 (Liu et al)",
                    xy=(2.4e-3, norm * 7e-8),
                    fontsize=10,
                    color="grey",
                    ha="left",
                    va="bottom",
                    rotation=0,
                )

            # ax.annotate(r'Beam dumps', xy=(1.2e-3, 0.8e-7), fontsize=10, color='grey', ha='left', va='bottom')
            ax.annotate(
                r"BaBar/Belle",
                xy=(0.4, norm * 1e-6),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
            )
            ax.annotate(
                r"SINDRUM",
                xy=(0.02, 0.5),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
            )
            ax.annotate(
                r"E141",
                xy=(3.8e-3, 3e-1),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=-36,
            )

    if c_lepton[0, 0] == 0 and c_lepton[1, 1] == 0 and c_lepton[2, 2] > 0:
        limits = [
            "E137_tauphilic.dat",
            "NA64mu_tauphilic.dat",
            "LEP_tauphilic.dat",
            "NA64_visible_tauphilic.dat",
            "NA64_invisible_tauphilic.dat",
        ]
        for limit in limits:
            ###########################################################
            x, y = np.genfromtxt("digitized/" + limit, unpack=True)
            x = x * 1e-3
            x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
            ax.fill(
                x,
                y,
                facecolor="lightgrey",
                edgecolor="None",
                linestyle="-",
                zorder=1.4,
                # label='E137 (Araki et al. 2021)',
            )
            ax.fill(
                x,
                y,
                facecolor="None",
                edgecolor=edgecolor,
                linestyle="-",
                zorder=2,
                linewidth=2 * linewidth,
                # label='E137 (Araki et al. 2021)',
            )

        # Add text labels to the gray regions
        if linewidth:
            ax.annotate(
                r"E137",
                xy=(1e-1, 2e-1),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=-50,
            )
            ax.annotate(
                r"NA64$\mu$ (vis)",
                xy=(2.2e-3, 0.7),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
            )
            ax.annotate(
                r"NA64e (inv)",
                xy=(2.2e-3, 1.2),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
            )
            ax.annotate(
                r"LEP",
                xy=(1, 3),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=0,
            )
            ax.annotate(
                r"NA64e (vis)",
                xy=(0.3, 6),
                fontsize=10,
                color="grey",
                ha="left",
                va="bottom",
                rotation=0,
            )


def make_Bvis_plot_LFV(
    BP_NAME, c_lepton, c_NN, mN, ma_fixed, smear=False, ymax=1e-3, ymin=1e-10
):

    fig, ax = std_fig(figsize=(5, 5))

    plot_other_limits_Bvis(
        ax, ma_fixed=ma_fixed, c_lepton=c_lepton, c_NN=c_NN, mN=mN, linewidth=0.0
    )

    # labels for legend
    labels = []
    labelnames = []
    name = BP_NAME

    Nsig = 2.3
    X, Y, Z = np.load(f"data/invfa_vs_Bvis_CHARM_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contourf(
        X,
        Y,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[0], 0.5)],
        alpha=1,
        zorder=1.5,
    )
    _ = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("CHARM")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_BEBC_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contourf(
        X,
        Y,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[1], 0.5)],
        alpha=1,
        zorder=1.4,
    )
    _ = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[1],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("BEBC")

    # X,Y,Z = np.load(f'data/NA62_rates_{name}.npy', allow_pickle=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('firebrick', 0.85)], alpha=1, zorder=1.1)
    # _ = ax.contour(X, Y, Z, levels=[Nsig], colors='firebrick', linestyles='-', linewidths=[1], alpha=1, zorder=2)
    # labels.append(c.legend_elements()[0][0])
    # labelnames.append(r'NA62 ($1.4\times 10^{17}$ POT)')

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_NoVA_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('darkorange', 0.5)], alpha=1, zorder=1.3)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles=[(1, (2, 1))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("NOvA")

    X, Y, Z = np.load(
        f"data/invfa_vs_Bvis_MicroBooNE_rates_{name}.npy", allow_pickle=True
    )
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('black', 0.5)], alpha=1, zorder=1.2)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[1],
        linestyles=[(1, (4, 1))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("$\mu$BooNE (NuMI)")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_ICARUS_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('black', 0.5)], alpha=1, zorder=1.1)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles=[(1, (6, 0))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("ICARUS (NuMI)")

    X1, Y1, Z1 = np.load(
        f"data/invfa_vs_Bvis_ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True
    )
    _, _, Z2 = np.load(
        f"data/invfa_vs_Bvis_ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True
    )
    X, Y, Z = X1, Y1, Z1 + Z2
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="lightgreen",
        linestyles=[(1, (3, 2))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"Proto-DUNE")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_FASER_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color='black', 0.95)], alpha=1, zorder=1.91)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(1, (2, 0.5))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("FASER")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_FASER2_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(1, (5, 1))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"FASER-2")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_SHiP_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles="-",
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"SHiP")

    ax.legend(
        labels,
        labelnames,
        loc="upper left",
        fontsize=8.5,
        ncol=1,
        frameon=True,
        framealpha=0.8,
        edgecolor="black",
        fancybox=False,
        handlelength=2.5,
        handletextpad=0.5,
        labelspacing=0.5,
        borderpad=0.5,
    )

    if c_lepton[0, 0] != c_lepton[0, 2]:
        title = r"{\bf LFV hierarchy} $\,\,\vert\,\,$"
        title += rf"$g_{{\ell \ell}} = {int(c_lepton[0,0])}$"
        title += r"$\,\,\vert\,\, g_{{(e,\mu)\tau}} = \lambda$"
        title += r"$\,\,\vert\,\, g_{{e\mu}} = \lambda^2 $"
        title += rf"$\,\,\vert\,\, \lambda = {sci_notation(c_lepton[1,2], notex=True, precision=1)}$"

    elif c_lepton[0, 0] == c_lepton[0, 2]:
        title = r"{\bf LFV anarchy} $\,\,\vert\,\,$"
        # ax.annotate(r'\noindent \bf LFV anarchy \\ $\tau$ limits only', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=11, ha='left', va='top')
        title += rf"$g_{{\ell_1 \ell_2}} = {int(c_lepton[0,0])}$"

    title += rf"$\,\vert\, m_a = {ma_fixed}$~GeV"

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$1/f_a$ [GeV$^{-1}$]")
    ax.set_xlabel(
        r"$\mathcal{B}(a \to {\rm vis}) = 1 - \mathcal{B}(a\to \text{dark sector})$"
    )
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1e-6, 1)

    ax.set_title(title, fontsize=11, pad=10)
    fig.savefig(f"plots/ALP_benchmark_{name}_Bvis.pdf", bbox_inches="tight", dpi=400)


def make_Bvis_plot_LFC(
    BP_NAME, c_lepton, c_NN, mN, ma_fixed, smear=False, ymax=1e-3, ymin=1e-10
):

    fig, ax = std_fig(figsize=(5, 5))

    plot_other_limits_Bvis(
        ax, ma_fixed=ma_fixed, c_lepton=c_lepton, c_NN=c_NN, mN=mN, linewidth=0.0
    )

    # labels for legend
    labels = []
    labelnames = []
    name = BP_NAME

    Nsig = 2.3
    X, Y, Z = np.load(f"data/invfa_vs_Bvis_CHARM_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contourf(
        X,
        Y,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[0], 0.5)],
        alpha=1,
        zorder=1.5,
    )
    _ = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("CHARM")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_BEBC_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contourf(
        X,
        Y,
        Z,
        levels=[Nsig, 1e100],
        colors=[lighten_color(CB_color_cycle_2[1], 0.5)],
        alpha=1,
        zorder=1.4,
    )
    _ = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[1],
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("BEBC")

    # X,Y,Z = np.load(f'data/NA62_rates_{name}.npy', allow_pickle=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('firebrick', 0.85)], alpha=1, zorder=1.1)
    # _ = ax.contour(X, Y, Z, levels=[Nsig], colors='firebrick', linestyles='-', linewidths=[1], alpha=1, zorder=2)
    # labels.append(c.legend_elements()[0][0])
    # labelnames.append(r'NA62 ($1.4\times 10^{17}$ POT)')

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_NoVA_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('darkorange', 0.5)], alpha=1, zorder=1.3)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles=[(1, (2, 1))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("NOvA")

    X, Y, Z = np.load(
        f"data/invfa_vs_Bvis_MicroBooNE_rates_{name}.npy", allow_pickle=True
    )
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('black', 0.5)], alpha=1, zorder=1.2)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[1],
        linestyles=[(1, (4, 1))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("$\mu$BooNE (NuMI)")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_ICARUS_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color('black', 0.5)], alpha=1, zorder=1.1)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors=CB_color_cycle_2[0],
        linestyles=[(1, (6, 0))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("ICARUS (NuMI)")

    X1, Y1, Z1 = np.load(
        f"data/invfa_vs_Bvis_ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True
    )
    _, _, Z2 = np.load(
        f"data/invfa_vs_Bvis_ProtoDUNE-NP02_rates_{name}.npy", allow_pickle=True
    )
    X, Y, Z = X1, Y1, Z1 + Z2
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="lightgreen",
        linestyles=[(1, (3, 2))],
        linewidths=[1.25],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"ProtoDUNE")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_FASER_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    # c = ax.contourf(X, Y, Z, levels=[Nsig, 1e100], colors=[lighten_color='black', 0.95)], alpha=1, zorder=1.91)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(1, (2, 0.5))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append("FASER")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_FASER2_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles=[(1, (5, 1))],
        linewidths=[1.5],
        alpha=1,
        zorder=2,
    )
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"FASER-2")

    X, Y, Z = np.load(f"data/invfa_vs_Bvis_SHiP_rates_{name}.npy", allow_pickle=True)
    X, Y, Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    c = ax.contour(
        X,
        Y,
        Z,
        levels=[Nsig],
        colors="black",
        linestyles="-",
        linewidths=[1.75],
        alpha=1,
        zorder=2,
    )
    # x,y = c.collections[0].get_paths()[0].vertices[:,0],c.collections[0].get_paths()[0].vertices[:,1]
    # # x=np.append([1e-2], x)
    # # y=np.append([1e-2], y)
    # x,y=plot_closed_region((x,y), logx=True, logy=True)
    # c = ax.plot(x,y, edgecolor='black', facecolor='None', linestyle='-', linewidth=1.75, alpha=1, zorder=2)
    # labels.append(c[0])
    labels.append(c.legend_elements()[0][0])
    labelnames.append(r"SHiP")

    ax.legend(
        labels,
        labelnames,
        loc="upper left",
        fontsize=8.5,
        ncol=1,
        frameon=True,
        framealpha=0.8,
        edgecolor="black",
        fancybox=False,
        handlelength=2.5,
        handletextpad=0.5,
        labelspacing=0.5,
        borderpad=0.5,
    )

    if c_lepton[0, 0] != c_lepton[0, 2]:
        title = r"{\bf LFV hierarchy} $\,\,\vert\,\,$"
        title += rf"$g_{{\ell \ell}} = {int(c_lepton[0,0])}$"
        title += rf"$\,\,\vert\,\, g_{{(e,\mu)\tau}} = \lambda$"
        title += rf"$\,\,\vert\,\, g_{{e\mu}} = \lambda^2 $"
        title += rf"$\,\,\vert\,\, \lambda = {sci_notation(c_lepton[1,2], notex=True, precision=1)}$"

    elif c_lepton[0, 0] == c_lepton[0, 2]:
        title = r"{\bf LFV anarchy} $\,\,\vert\,\,$"
        # ax.annotate(r'\noindent \bf LFV anarchy \\ $\tau$ limits only', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=11, ha='left', va='top')
        title += rf"$g_{{\ell_1 \ell_2}} = {int(c_lepton[0,0])}$"

    title += rf"$\,\vert\, m_a = {ma_fixed}$~GeV"

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$1/f_a$ [GeV$^{-1}$]")
    ax.set_xlabel(
        r"$\mathcal{B}(a \to {\rm vis}) = 1 - \mathcal{B}(a\to \text{dark sector})$"
    )
    ax.set_ylim(ymin, ymax)
    # ax.set_xlim(X.min(), 1)
    ax.set_xlim(1e-6, 1)

    ax.set_title(title, fontsize=11, pad=10)
    # ax.invert_xaxis()
    # ax.set_xticklabels(ax.get_xticklabels()[::-1])
    fig.savefig(f"plots/ALP_benchmark_{name}_Bvis.pdf", bbox_inches="tight", dpi=400)


def plot_hist_with_errors(
    ax,
    data,
    weights,
    bins,
    label,
    color,
    zorder=2,
    lw=1.5,
    nevents=1,
    ls="-",
    normalize=False,
    alpha=0.5,
    histtype="step",
    density=False,
):
    if normalize:
        norm = np.max(weights)
    else:
        norm = 1
    weights = weights / norm
    # Compute histogram and sum of squared weights per bin
    counts, bin_edges = np.histogram(data, bins=bins, weights=weights * nevents)
    sumw2, _ = np.histogram(data, bins=bins, weights=np.square(weights * nevents))

    # Bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_widths = np.diff(bin_edges)

    counts /= nevents
    sumw2 /= nevents**2
    # Plot main histogram
    ax.hist(
        data,
        bins=bins,
        weights=weights * norm,
        label=label,
        histtype=histtype,
        edgecolor=color,
        facecolor=color,
        linestyle=ls,
        density=density,
        zorder=zorder,
        lw=lw,
    )

    # Plot error bars as bars around each bin
    if density:
        norm /= bin_widths
    ax.bar(
        bin_centers,
        2 * np.sqrt(sumw2) * norm,
        bottom=(counts - np.sqrt(sumw2)) * norm,
        width=bin_widths,
        edgecolor="None",
        facecolor=color,
        alpha=alpha,
        lw=0,
        zorder=zorder - 0.1,
    )


def make_ctau_plot(
    inv_fa_range=[0.5e-9, 1e-4],
    ma_range=[1e-2, const.m_tau - const.m_e * 1.01],
    Npoints=101,
    c_lepton=False,
    name="",
    c_NN=0,
    mN=0,
):

    fig, ax = std_fig(figsize=(5, 5))
    inv_fas = np.geomspace(*inv_fa_range, Npoints, endpoint=True)
    m_alps = np.geomspace(*ma_range, Npoints, endpoint=True)
    MA, INV_FA = np.meshgrid(m_alps, inv_fas)

    alp = ALP(MA, 1 / INV_FA, c_NN=c_NN, mN=mN, Bvis=1)
    Z = const.get_decay_rate_in_cm(alp.Gamma_a)
    print(Z.min(), Z.max())
    ax.contour(
        MA,
        INV_FA,
        np.log10(Z),
        levels=100,
        cmap="Blues",
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )

    ax.set_yscale("log")
    ax.set_xscale("log")


def make_ctau_plot_Bvis(
    inv_fa_range=[0.5e-9, 1e-4],
    Bvis_range=[1e-12, 1],
    Npoints=101,
    c_lepton=False,
    name="",
    c_NN=0,
    mN=0,
    ma_fixed=0.3,
):

    fig, ax = std_fig(figsize=(5, 5))
    inv_fas = np.geomspace(*inv_fa_range, Npoints, endpoint=True)
    Bvis = np.geomspace(*Bvis_range, Npoints, endpoint=True)
    X, Y = np.meshgrid(Bvis, inv_fas)
    alp = ALP(ma_fixed, 1 / Y, c_NN=c_NN, mN=mN, Bvis=X)
    Z = const.get_decay_rate_in_cm(alp.Gamma_a)
    # X,Y,Z = interp_grid(X.flatten(), Y.flatten(), Z.flatten(), logx=True, logy=True)
    ax.contour(
        X,
        Y,
        np.log10(Z),
        levels=20,
        cmap="Blues",
        linestyles="-",
        linewidths=[1],
        alpha=1,
        zorder=2,
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
