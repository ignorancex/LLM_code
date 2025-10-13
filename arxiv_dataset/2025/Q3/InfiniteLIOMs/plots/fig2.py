import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import matplotlib.ticker as ticker

filename_spin_op = (
    lambda M: f"../data/XXX_QC/eigenvalues_XXX_QC_M_{M}_J_1.0_d_0.5_hermitian_P_even_Sz_cons_true_spin_op_basis.txt"
)

filename_evec = (
    lambda M: f"../data/XXX_QC/eigenvectors_XXX_QC_M_{M}_J_1.0_d_0.5_hermitian_P_even_Sz_cons_true_spin_op_basis.txt"
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
legend_fontsize = 12
label_fontsize = 15
text_fontsize = 12
tick_fontsize = 14

fig, axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(6.3,3),
    gridspec_kw={"width_ratios": [1,1]},
    constrained_layout=False,
)


ax = axs[0]
pauli_M = [3, 4, 5]
markers = ["o", "s","^"]
colors = plt.cm.viridis(np.linspace(0, 1, 2*len(pauli_M)))
colors = [colors[0], colors[2], colors[4]]


data = np.loadtxt(filename_spin_op(3))
N3 = np.shape(data)[0]
data = np.loadtxt(filename_spin_op(4))
N4 = np.shape(data)[0]
data = np.loadtxt(filename_spin_op(5))
N5 = np.shape(data)[0]  

print(N3, N4, N5)

for i, M in enumerate(pauli_M):
    data = np.loadtxt(filename_spin_op(M))
    idx = np.arange(len(data)) + 1
    ax.plot(
        idx,
        data,
        label=fr"$M={{{M}}}$",
        linewidth=0.5,
        marker=markers[i],
        markersize=7,
        color=colors[i],
        linestyle="--",
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
ax.set_xlim(0.5, 9.5)
ax.set_ylim(-0.05, 1.5)
ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
ax.set_xlabel(r"$\alpha$", fontsize=label_fontsize)
ax.set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)
ax.text(
    0.1,
    0.5,
    r"\textbf{(a)}",
    fontsize=text_fontsize,
    transform=ax.transAxes,
)
ax.legend(loc="upper left", fontsize=legend_fontsize, frameon=False)
ax.set_xticks(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$", r"$9$"],
)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="y", which="minor", length=2, color="black")


ax = axs[1]
markers = ["o", "s","^"]
colors = plt.cm.viridis(np.linspace(0, 1, 4))
colors = [colors[2*i] for i in range(2)]

data = np.loadtxt(filename_evec(4))
idx = np.arange(np.shape(data)[0]) + 1
ax.bar(
    idx,
    data[:, 0]**2 + data[:, 1]**2,
    label=fr"$M=4$",
    color=colors[1],
    alpha=0.75,
    zorder = 3,
)
data = np.loadtxt(filename_evec(5))
idx = np.arange(np.shape(data)[0]) + 1
ax.bar(
    idx,
    data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2,
    label=fr"$M=5$",
    color=colors[0],
    alpha=0.75,
)
ax.axvline(N3+0.5, color="black", linewidth=1.5, linestyle="--", zorder=10)
ax.axvline(N4+0.5, color="black", linewidth=1.5, linestyle="--", zorder=10)
ax.set_xlim(0.0, N5)
ax.set_ylim(1e-6, 1.0)
ax.set_xlabel(r"$s$", fontsize=label_fontsize)
ax.set_ylabel(r"Mazur bound $B_s$", fontsize=label_fontsize)
ax.text(
    0.4,
    0.85,
    r"\textbf{(b)}",
    fontsize=text_fontsize,
    transform=ax.transAxes,
)

ax.text(
  0.018,
  0.1,
  r"$m \leq 3$",
  fontsize=text_fontsize-1,
  rotation=90,
  transform=ax.transAxes,
  color="black",
  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
  zorder=10,
)
ax.text(
  0.16,
  0.1,
  r"$m = 4$",
  fontsize=text_fontsize-1,
  transform=ax.transAxes,
  color="black",
  rotation=90,
  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
  zorder=10,
)
ax.text(
  0.38,
  0.1,
  r"$m = 5$",
  fontsize=text_fontsize-1,
  transform=ax.transAxes,
  color="black",
  rotation=90,
  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
  zorder=10,
)

ax.legend(loc="upper right", fontsize=legend_fontsize-2, frameon=False)
ax.set_xticks(
    [1, 20, 40, 60, 80, 100, 120, 140, 160, 180],
    [r"$1$", r"", r"$40$", r"", r"$80$", r"", r"$120$", r"", r"$160$", r""],
    minor=False,
)
ax.set_xticks(
  [5, 10, 15, 25, 30, 35, 45, 50, 55, 65, 70, 75, 85, 90, 95, 105, 110, 115, 125, 130, 135, 145, 150, 155, 165, 170, 175],
  minor=True)
ax.set_xticklabels([], minor=True)

ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="y", which="minor", length=2, color="black")
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("fig2.pdf", dpi=300, bbox_inches="tight")