import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
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
text_fontsize = 12
tick_fontsize = 14
width = 1

fig = plt.figure(figsize=(9.2, 3.3), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[1, 20])

ax4 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1,  0])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[1, 2])

colors = plt.cm.viridis(np.linspace(0, 1, 7))
colors = [colors[0], colors[2], colors[4], colors[6]]



# ==================== add data from ladder paper ====================

data = np.loadtxt("../data/spin_ladder/Qminus_U_scaling.csv", delimiter=",")
ax2.plot(
    data[:, 0],
    data[:, 1],
    marker="",
    markersize=6,
    color=colors[2],
    label=r"$\Gamma^-$",
    linestyle="--",
    linewidth=3,
    zorder = 10,
)
data = np.loadtxt("../data/spin_ladder/Qminus_delta_scaling.csv", delimiter=",")
ax3.plot(
    data[:, 0],
    data[:, 1],
    marker="",
    markersize=6,
    color=colors[2],
    label=r"$\Gamma^-$",
    linestyle="--",
    linewidth=3,
    zorder = 10,
)


data = np.loadtxt("../data/spin_ladder/spin_ladder_U_delta_eigenvalues.txt", delimiter=" ")

# =========================== Panel (b): lambda1 and lambda2 vs U for delta=0.3 ==================
# Use the original data
deltas = data[:, 0]
Us = data[:, 1]
lambda1 = data[:, 2]
lambda2 = data[:, 3]

# Select delta=0.3 (allow for floating point tolerance)
delta_target = 0.3
tol = 1e-10
mask = np.abs(deltas - delta_target) < tol

Us_sel = Us[mask]
lambda1_sel = lambda1[mask]
lambda2_sel = lambda2[mask]

# Sort by U for plotting
sort_idx = np.argsort(Us_sel)
Us_sel = Us_sel[sort_idx]
lambda1_sel = lambda1_sel[sort_idx]
lambda2_sel = lambda2_sel[sort_idx]

# Plot lambda1 vs U
ax2.plot(
    Us_sel,
    lambda1_sel,
    marker="o",
    markersize=6,
    color=colors[1],
    label=r"$\alpha=1$",
    linestyle="",
)

# Plot lambda2 vs U
ax2.plot(
    Us_sel,
    lambda2_sel,
    marker="s",
    markersize=6,
    color=colors[3],
    label=r"$\alpha=2$",
    # markeredgecolor="black",
    # markeredgewidth=0.2,
    linestyle="",
)

# Fit a line to log-log data for lambda1 vs U
log_Us_sel = np.log10(Us_sel[1:20])
log_lambda1_sel = np.log10(lambda1_sel[1:20])
fit_coeffs2 = np.polyfit(log_Us_sel, log_lambda1_sel, 1)
xxx = np.logspace(-3, 1, 100)
fit_line2 = np.polyval(fit_coeffs2, np.log10(xxx))

ax2.plot(
    xxx,
    10**fit_line2,
    color="black",
    linestyle="-",
    linewidth=2,
    zorder=1,
    # label=fr"Fit: slope={fit_coeffs2[0]:.2f}"
)

txtlab2 = r"$\propto U^{%1.f}$" % fit_coeffs2[0]
ax2.text(
    0.15,
    0.20,
    txtlab2,
    fontsize=text_fontsize+2,
    color="black",
    rotation=30,
    transform=ax2.transAxes,
)

ax2.set_xlabel(r"$U$", fontsize=label_fontsize)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylim(1e-7, 1)
ax2.set_xlim(7e-3, 1)
ax2.set_yticks(
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
ax2.set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)
ax2.yaxis.set_minor_locator(
    mpl.ticker.LogLocator(base=10, subs=np.arange(1, 10) * 0.1, numticks=100)
)
ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax2.legend(fontsize=text_fontsize, loc="lower right", frameon=False)
ax2.tick_params(axis="both", which="major", labelsize=tick_fontsize)
ax2.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
ax2.text(0.1, 0.85, r"(b) $\Delta=0.3$", fontsize=text_fontsize, transform=ax2.transAxes)

# ================== Panel (c): lambda1 and lambda2 vs delta for U=0.3 ==================

# Use the original data
deltas = data[:, 0]
Us = data[:, 1]
lambda1 = data[:, 2]
lambda2 = data[:, 3]

# Select U=0.3 (allow for floating point tolerance)
U_target = 0.3
tol = 1e-10
mask = np.abs(Us - U_target) < tol

deltas_sel = deltas[mask]
lambda1_sel = lambda1[mask]
lambda2_sel = lambda2[mask]

# Sort by delta for plotting
sort_idx = np.argsort(deltas_sel)
deltas_sel = deltas_sel[sort_idx]
lambda1_sel = lambda1_sel[sort_idx]
lambda2_sel = lambda2_sel[sort_idx]

# Plot lambda1 vs delta
ax3.plot(
    deltas_sel,
    lambda1_sel,
    marker="o",
    markersize=6,
    color=colors[1],
    label=r"$\alpha=1$",
    linestyle="",
)

# Plot lambda2 vs delta
ax3.plot(
    deltas_sel,
    lambda2_sel,
    marker="s",
    markersize=6,
    color=colors[3],
    label=r"$\alpha=2$",
    linestyle="",
)

# Fit a line to log-log data for lambda1 vs delta
log_deltas_sel = np.log10(deltas_sel[1:20])
log_lambda1_sel = np.log10(lambda1_sel[1:20])
fit_coeffs3 = np.polyfit(log_deltas_sel, log_lambda1_sel, 1)
xxx = np.logspace(-3, 1, 100)
fit_line3 = np.polyval(fit_coeffs3, np.log10(xxx))
ax3.plot(
    xxx,
    10**fit_line3,
    color="black",
    linestyle="-",
    linewidth=2,
    zorder=1,
    # label=fr"Fit: slope={fit_coeffs3[0]:.2f}"
)

txtlab3 = r"$\propto \Delta^{%1.f}$" % fit_coeffs3[0]
ax3.text(
    0.15,
    0.20,
    txtlab3,
    fontsize=text_fontsize+2,
    color="black",
    rotation=30,
    transform=ax3.transAxes,
)

ax3.set_xlabel(r"$\Delta$", fontsize=label_fontsize)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylim(1e-7, 1)
ax3.set_xlim(7e-3, 1)
ax3.set_yticks(
    [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    [
        r"",
        r"",
        r"",
        r"",
        r"",
        r"",
        r"",
    ],
)
ax3.yaxis.set_minor_locator(
    mpl.ticker.LogLocator(base=10, subs=np.arange(1, 10) * 0.1, numticks=100)
)
ax3.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax3.legend(fontsize=text_fontsize, loc="lower right", frameon=False)
ax3.tick_params(axis="both", which="major", labelsize=tick_fontsize)
ax3.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
ax3.text(0.1, 0.85, r"(c) $U=0.3$", fontsize=text_fontsize, transform=ax3.transAxes)


# Panel (a): Second eigenvalue
deltas = data[:, 0]
Us = data[:, 1]
lambdas = data[:, 3]

mask = lambdas >= 1e-10
lambdas = lambdas[mask]
deltas = deltas[mask]
Us = Us[mask]
deltas = np.unique(deltas)
Us = np.unique(Us)
lambdas = lambdas.reshape(len(deltas), len(Us))

data2 = np.transpose(lambdas)
# data2 = np.sqrt(data2)
levels = np.logspace(np.log10(data2.min()), np.log10(data2.max()), 15)
levels = levels[:-3]

norm = mpl.colors.LogNorm(vmin=1e-6, vmax=1e-1)
cmap = cm.viridis_r
level_color = "white"
im2 = ax1.pcolormesh(
    deltas, Us, data2, cmap=cmap, norm=norm, linewidth=0, rasterized=True
)
CS2 = ax1.contour(deltas, Us, data2, levels, colors=level_color, linewidths=width)
ax1.set_xlabel(r"Anisotropy $\Delta$", fontsize=label_fontsize)
ax1.set_ylabel(r"Interchain coupling $U$", fontsize=label_fontsize)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(1e-2, 1)
ax1.set_ylim(1e-2, 1)
# ax1.set_yticklabels([])
ax1.text(
    0.1,
    0.85,
    r"\textbf{(a)} $\lambda_2$",
    fontsize=text_fontsize,
    ha="left",
    transform=ax1.transAxes,
    color="white",
)
ax1.tick_params(axis="both", which="major", labelsize=tick_fontsize)
ax1.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

# colorbar
cbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax4,
    pad=0.03,
    shrink=0.88,
    aspect=20,
    location="top",
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
    fontsize = tick_fontsize-1,
)

cbar.ax.minorticks_on()
cbar.ax.xaxis.set_tick_params(which="minor", length=2, width=1, labelsize=tick_fontsize)
cbar.ax.xaxis.set_tick_params(which="major", length=4, width=1, labelsize=tick_fontsize-2)

# Add horizontal and vertical red lines on ax3
# Example: vertical at delta=0.1, horizontal at lambda=1e-4
ax1.axvline(x=0.3, color='red', linestyle='--', linewidth=1)
ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=1)

plt.savefig("figS3.pdf", bbox_inches="tight")
