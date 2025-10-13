import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm


def latex_plot(scale=1, fontsize=12):
    """Changes the size of a figure and fonts for the publication-quality plots."""
    fig_width_pt = 246.0
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    eps_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": fig_size,
    }
    mpl.rcParams.update(eps_with_latex)


latex_plot(scale=1, fontsize=12)

label_fontsize = 18
text_fontsize = 14
tick_fontsize = 14
width = 1
level_color = "white"
norm = mpl.colors.LogNorm(vmin=1e-6, vmax=1e-1)
cmap = cm.viridis_r

fig = plt.figure(figsize=(7, 3.3), constrained_layout=True)
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 20], figure=fig)

ax4 = fig.add_subplot(gs[0, 1])  # top right
ax1 = fig.add_subplot(gs[1, 1])  # top left
ax2 = fig.add_subplot(gs[1, 0])  # bottom left
ax3 = fig.add_subplot(gs[0, 0])  # right column spanning both rows

ax4.axis('off')

colors = plt.cm.viridis(np.linspace(0, 1, 7))
colors = [colors[0], colors[2], colors[4], colors[6]]

# ==================== add data from ladder paper ====================

# Delta U
data = np.loadtxt("../data/spin_ladder/Qplus_scaling.csv", delimiter=",")

ax1.plot(
    data[:, 0],
    data[:, 1],
    marker="",
    markersize=6,
    color=colors[2],
    label=r"$\Gamma^+$",
    linestyle="--",
    linewidth=3,
    zorder = 10,
)

data = np.loadtxt("../data/spin_ladder/spin_ladder_U_delta_eigenvalues.txt", delimiter=" ")
# ============================ Panel (a): diagonal scaling ===========================
deltas = data[:, 0]
Us = data[:, 1]
lambdas = data[:, 2]

mask = lambdas >= 1e-10
lambdas = lambdas[mask]
deltas = deltas[mask]
Us = Us[mask]
deltas = np.unique(deltas)
Us = np.unique(Us)
lambdas = lambdas.reshape(len(deltas), len(Us))

data1 = lambdas
diag_indices = np.arange(min(len(deltas), len(Us)))
lambda_diag = data1[diag_indices, diag_indices]
deltaU_diag = deltas[diag_indices] * Us[diag_indices]

# Fit a line to log-log data
log_deltaU_diag = np.log10(deltaU_diag)
log_lambda_diag = np.log10(lambda_diag)
fit_coeffs = np.polyfit(log_deltaU_diag[:20], log_lambda_diag[:20], 1)
fit_line = np.polyval(fit_coeffs, log_deltaU_diag)
ax1.plot(
    deltaU_diag,
    10**fit_line,
    color=colors[0],
    linestyle="-",
    linewidth=2,
)
ax1.legend(fontsize=text_fontsize)
# Plot diagonal values vs delta*U
ax1.plot(
    deltaU_diag,
    lambda_diag,
    marker="o",
    markersize=6,
    color=colors[1],
    label=rf"$\lambda_1$",
    linestyle="",
)


ax1.set_xlabel(r"$\Delta U$", fontsize=label_fontsize)
ax1.set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(3e-4, 1)
ax1.set_ylim(1e-7, 1)
ax1.set_xticks(
    [1e-3, 1e-2, 1e-1, 1], [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"]
)
ax1.xaxis.set_minor_locator(
    mpl.ticker.LogLocator(base=10, subs=np.arange(1, 10) * 0.1, numticks=100)
)
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.set_yticks(
    [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    [
        r"$10^{-7}$",
        r"$10^{-6}$",
        r"$10^{-5}$",
        r"$10^{-4}$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
    ],
)
ax1.yaxis.set_minor_locator(
    mpl.ticker.LogLocator(base=10, subs=np.arange(1, 10) * 0.1, numticks=100)
)
ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.legend(fontsize=text_fontsize, loc="lower right", frameon=False)
ax1.tick_params(axis="both", which="major", labelsize=tick_fontsize)
ax1.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

txtlab1 = r"$\propto\left(\Delta U \right)^{%1.f}$" % fit_coeffs[0]
ax1.text(
    0.06,
    0.2,
    txtlab1,
    fontsize=text_fontsize + 2,
    color=colors[0],
    rotation=43,
    transform=ax1.transAxes,
)
ax1.text(0.1, 0.9, r"\textbf{(b)}", fontsize=text_fontsize, transform=ax1.transAxes)


# Panel (b): First eigenvalue
deltas = data[:, 0]
Us = data[:, 1]
lambdas = data[:, 2]

mask = lambdas >= 1e-10
lambdas = lambdas[mask]
deltas = deltas[mask]
Us = Us[mask]
deltas = np.unique(deltas)
Us = np.unique(Us)
lambdas = lambdas.reshape(len(deltas), len(Us))

data1 = np.transpose(lambdas)
# data1 = np.sqrt(data1)
levels = np.logspace(np.log10(data1.min()), np.log10(data1.max()), 15)
levels = levels[:-3]
im1 = ax2.pcolormesh(
    deltas, Us, data1, cmap=cmap, norm=norm, linewidth=0, rasterized=True
)
CS1 = ax2.contour(deltas, Us, data1, levels, colors=level_color, linewidths=width)
ax2.set_ylabel(r"Interchain coupling $U$", fontsize=label_fontsize)
ax2.set_xlabel(r"Anisotropy $\Delta$", fontsize=label_fontsize)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(1e-2, 1)
ax2.set_ylim(1e-2, 1)
ax2.tick_params(axis="both", which="major", labelsize=tick_fontsize)
ax2.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

ax2.text(
    0.64,
    0.9,
    r"\textbf{(a)} $\lambda_1$",
    fontsize=text_fontsize,
    ha="left",
    transform=ax2.transAxes,
    color="white",
)

# Plot a red diagonal line on ax1 (panel b)
min_val = max(ax2.get_xlim()[0], ax2.get_ylim()[0])
max_val = min(ax2.get_xlim()[1], ax2.get_ylim()[1])
ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)

# Panel (b): Second eigenvalue
# colorbar

cbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax3,
    pad=0.03,
    shrink=0.88,
    aspect=20,
    location="top"
)
cbar.set_ticks([1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1])
cbar.set_ticklabels(
    [
        r"$\leq 10^{-6}$",
        r"$10^{-5}$",
        r"$10^{-4}$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$\geq 10^{-1}$",
    ],
    fontsize = tick_fontsize-2,
)

cbar.ax.minorticks_on()
cbar.ax.xaxis.set_tick_params(which="minor", length=2, width=1, labelsize=tick_fontsize)
cbar.ax.xaxis.set_tick_params(which="major", length=4, width=1, labelsize=tick_fontsize-2)

plt.savefig("fig3.pdf", bbox_inches="tight")
