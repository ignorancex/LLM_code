import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

filename_spin_op = (
    lambda M, d, U : f"../data/spin_ladder/eigenvalues_spin_ladder_M_{M}_J_1.0_d_{d}_U_{U}_T_imag_P_even_Sz_cons_true_spin_op_basis.txt"
)


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
legend_fontsize = 9
label_fontsize = 15
text_fontsize = 10
tick_fontsize = 14

fig, axs = plt.subplots(1, 2, figsize=(6, 3))

# Two XXZ chains
M_spin_op = [2,3,4,5]
colors = plt.cm.viridis(np.linspace(0, 1, len(M_spin_op)))
markers = ["o", "D", "v", ">", "d"]

for i, M in enumerate(M_spin_op):

    data = np.loadtxt(filename_spin_op(M, "0.0", "0.3"))
    idx = np.arange(len(data)) + 1
    axs[0].plot(
        idx,
        data,
        label=fr"$M={{{M}}}$",
        linewidth=0.5,
        marker=markers[i],
        markersize=5,
        color=colors[i],
        linestyle="--",
        markeredgecolor="black",
        markeredgewidth=0.1,
        zorder = 100 - i,
    )


axs[0].set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)
axs[0].set_xlim(0.5, 8.5)
axs[0].set_ylim(-0.02, 0.4)
axs[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[0].set_xlabel(r"$\alpha$", fontsize=label_fontsize)
axs[0].text(
    0.25,
    0.85,
    r"\textbf{(a)} Hubbard",
    fontsize=text_fontsize,
    ha="left",
    transform=axs[0].transAxes,
)
# axs[1].legend(loc="upper left", fontsize=legend_fontsize, frameon=False, ncol=2)
axs[0].set_xticks(
    [1, 2, 3, 4, 5, 6, 7, 8],
    [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$"],
)
axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[0].tick_params(axis="y", which="minor", length=2, color="black")

axs[0].legend(
    loc="best",
    fontsize=legend_fontsize,
    frameon=False,
    ncol=1,
)


# Two XXZ chains
M_spin_op = [2,3,4,5]
colors = plt.cm.viridis(np.linspace(0, 1, len(M_spin_op)))
markers = ["o", "D", "v", ">", "d"]

for i, M in enumerate(M_spin_op):

    data = np.loadtxt(filename_spin_op(M, "0.3", "0.0"))
    idx = np.arange(len(data)) + 1
    axs[1].plot(
        idx,
        data,
        label=fr"$M={{{M}}}$",
        linewidth=0.5,
        marker=markers[i],
        markersize=5,
        color=colors[i],
        linestyle="--",
        markeredgecolor="black",
        markeredgewidth=0.1,
        zorder = 100 - i,
    )


axs[1].set_xlim(0.5, 8.5)
axs[1].set_ylim(-0.02, 0.4)
axs[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[1].set_xlabel(r"$\alpha$", fontsize=label_fontsize)
axs[1].text(
    0.25,
    0.85,
    r"\textbf{(b)} Two XXZ chains",
    fontsize=text_fontsize,
    ha="left",
    transform=axs[1].transAxes,
)
# axs[1].legend(loc="upper left", fontsize=legend_fontsize, frameon=False, ncol=2)
axs[1].set_xticks(
    [1, 2, 3, 4, 5, 6, 7, 8],
    [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$"],
)
axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[1].tick_params(axis="y", which="minor", length=2, color="black")

axs[1].legend(
    loc="best",
    fontsize=legend_fontsize,
    frameon=False,
    ncol=1,
)


plt.tight_layout()
plt.savefig("figC1.pdf", bbox_inches="tight")
