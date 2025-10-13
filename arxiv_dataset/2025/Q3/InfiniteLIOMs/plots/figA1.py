import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

filename_pauli = (
    lambda M, d=0.75, J=1.0: f"../data/1D_XXZ/eigenvectors_1D_XXZ_M_{M:d}_J_{J:.1f}_d_{d:.2f}_pauli_basis.txt"
)
filename_spin_op = (
    lambda M, T, P, d=0.75, J=1.0: f"../data/1D_XXZ/eigenvectors_1D_XXZ_M_{M:d}_J_{J:.1f}_d_{d:.2f}_T_{T}_P_{P}_Sz_cons_true_spin_op_basis.txt"
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

ax = axs[0]
markers = ["o", "s", "^"]

Ms = [2, 3]
colors = plt.cm.viridis(np.linspace(0, 1, 2 * len(Ms)))
colors = [colors[2 * i] for i in range(len(Ms))]

Ns = []

for i, M in enumerate(Ms):
    data = np.loadtxt(filename_pauli(M))
    drude = np.sum(np.square(data[:, 0:M]), axis=1)
    idx = np.arange(np.shape(data)[0]) + 1
    Ns.append(np.shape(data)[0])
    if i == 0:
        p1 = ax.scatter(
            idx[:2],
            drude[:2],
            label=rf"$M={M}$",
            color=colors[i],
            zorder=12,
            marker="v",
            s=20,
            clip_on=False,
        )
        p2 = ax.scatter(
            idx[3:],
            drude[3:],
            label=rf"$M={M}$",
            color=colors[i],
            zorder=12,
            marker="s",
            s=20,
            clip_on=False,
        )
    if i == 1:
        p3 = ax.bar(
            idx,
            drude,
            label=rf"$M={M}$",
            color=colors[i],
            alpha=0.5,
            zorder=3,
        )

for i, M in enumerate(Ms):
    ax.axvline(Ns[i] + 0.5, color="black", linewidth=1.5, linestyle="--", zorder=10)

ax.set_xlim(0.0, max(Ns) + 1)
ax.set_ylim(1e-6, 5.0)
ax.set_xlabel(r"$s$", fontsize=label_fontsize)
ax.set_ylabel(r"Mazur bound $B_s$", fontsize=label_fontsize)
ax.text(
    0.3,
    0.88,
    r"\textbf{(a)} Pauli",
    fontsize=text_fontsize,
    transform=ax.transAxes,
)

ax.legend(
    [(p1, p2), p3],
    [r"$M=2$", r"$M=3$"],
    loc="upper right",
    fontsize=legend_fontsize,
    frameon=False,
    handler_map = {tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
)

ax.set_xticks(
    [1, 10, 20, 30, 40, 50],
    [r"$1$", r"$10$", r"$20$", r"$30$", r"$40$", r"$50$"],
    minor=False,
)
ax.set_xticks(
  np.arange(1, 51, 1),
  minor=True)


ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="y", which="minor", length=2, color="black")
ax.set_yscale("log")

# Symmetry resolved basis
ax = axs[1]
markers = ["o", "s", "^"]

Ms = [2, 4]
colors = plt.cm.viridis(np.linspace(0, 1, 2 * len(Ms)))
colors = [colors[2 * i] for i in range(len(Ms))]

Ns = []

for i, M in enumerate(Ms):
    data = np.loadtxt(filename_spin_op(M, "real", "even"))
    drude = np.sum(np.square(data[:, 0 : (M // 2)]), axis=1)
    idx = np.arange(np.shape(data)[0]) + 1
    Ns.append(np.shape(data)[0])
    if i == 0:
        p1 = ax.scatter(
            idx[:2],
            drude[:2],
            label=rf"$M={M}$",
            color=colors[i],
            zorder=12,
            marker="s",
            s=20,
            clip_on=False,
        )
    if i == 1:
        p3 = ax.bar(
            idx,
            drude,
            label=rf"$M={M}$",
            color=colors[i],
            alpha=0.5,
            zorder=3,
        )

for i, M in enumerate(Ms):
    ax.axvline(Ns[i] + 0.5, color="black", linewidth=1.5, linestyle="--", zorder=10)

ax.set_xlim(0.0, max(Ns) + 1)
ax.set_ylim(1e-6, 5.0)
ax.set_xlabel(r"$s$", fontsize=label_fontsize)
ax.text(
  0.2,
  0.88,
  "\\textbf{(b)}",
  fontsize=text_fontsize,
  transform=ax.transAxes,
  multialignment='right',
)
ax.text(
  0.32,
  0.85,
  "Symmetry\nresolved",
  fontsize=text_fontsize,
  transform=ax.transAxes,
  multialignment='center',
)

ax.legend(
    loc="upper right",
    fontsize=legend_fontsize,
    frameon=False,
)

ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="x", which="minor", length=2, color="black")

ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="y", which="minor", length=2, color="black")
ax.set_yscale("log")

ax.set_yticklabels([])


plt.tight_layout()
plt.savefig("figA1.pdf", bbox_inches="tight")
