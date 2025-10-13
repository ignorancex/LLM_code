import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

filename_pauli = (
    lambda M, d=0.75, J=1.0: f"../data/1D_XXZ/eigenvalues_1D_XXZ_M_{M:d}_J_{J:.1f}_d_{d:.2f}_pauli_basis.txt"
)
filename_spin_op = (
    lambda M, T, P, d=0.75, J=1.0: f"../data/1D_XXZ/eigenvalues_1D_XXZ_M_{M:d}_J_{J:.1f}_d_{d:.2f}_T_{T}_P_{P}_Sz_cons_true_spin_op_basis.txt"
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

# Panel (a): Pauli basis
pauli_M = [2, 3, 4, 5, 6, 7, 8]
markers = ["o", "s", "D", "^", "v", "<", ">"]
colors = plt.cm.viridis(np.linspace(0, 1, len(pauli_M)))

for i, M in enumerate(pauli_M):
    data = np.loadtxt(filename_pauli(M))
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
        markeredgewidth=0.5,
    )
axs[0].set_xlim(0.5, 9.5)
axs[0].set_ylim(-0.2, 4)
axs[0].axhline(0, color="black", linewidth=0.5, linestyle="-")
axs[0].set_xlabel(r"$\alpha$", fontsize=label_fontsize)
axs[0].set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)
axs[0].text(
    0.05,
    0.3,
    r"\textbf{(a)} Pauli",
    fontsize=text_fontsize,
    transform=axs[0].transAxes,
)
axs[0].legend(loc="upper left", fontsize=legend_fontsize, frameon=False)
axs[0].set_xticks(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$", r"$9$"],
)
axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[0].tick_params(axis="y", which="minor", length=2, color="black")


# symmetry resolved basis
M_spin_op = [2,4,6,8,10]
colors = plt.cm.viridis(np.linspace(0, 1, len(M_spin_op)))
markers = ["o", "D", "v", ">", "d"]

for i, M in enumerate(M_spin_op):
    P = "even"
    if M % 2 == 0:
      T = "real"
    else:
      T = "imag"

    data = np.loadtxt(filename_spin_op(M, T, P))
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


axs[1].set_xlim(0.5, 9.5)
axs[1].set_ylim(-0.02, 0.5)
axs[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[1].set_xlabel(r"$\alpha$", fontsize=label_fontsize)
axs[1].text(
    0.35,
    0.45,
    r"\textbf{(b)} Symmetry resolved",
    fontsize=text_fontsize,
    ha="left",
    transform=axs[1].transAxes,
)
# axs[1].legend(loc="upper left", fontsize=legend_fontsize, frameon=False, ncol=2)
axs[1].set_xticks(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$", r"$9$"],
)
axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[1].tick_params(axis="y", which="minor", length=2, color="black")

axs[1].legend(
    loc="upper right",
    fontsize=legend_fontsize,
    frameon=False,
    ncol=1,
)


plt.tight_layout()
plt.savefig("fig1.pdf", bbox_inches="tight")
