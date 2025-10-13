import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.cm import viridis
import matplotlib.ticker as ticker
import os
import re


def reorder_handles(handles, labels):
    # You can reorder manually if needed
    col1 = handles[:3]
    col2 = handles[3:]
    new_handles = []
    new_labels = []
    max_len = max(len(col1), len(col2))
    for i in range(max_len):
        if i < len(col1):
            new_handles.append(col1[i])
            new_labels.append(labels[i])
        if i < len(col2):
            new_handles.append(col2[i])
            new_labels.append(labels[len(col1) + i])
    return new_handles, new_labels


path_to_drude1 = lambda M, d: os.path.join(
    "../data/spin_current_drude/other",
    f"drude_1D_XXZ_M_{M:d}_J_1.0_d_{d}_T_imag_P_odd_Sz_cons_true_spin_op_basis.txt",
)
path_to_drude2 = lambda M, l, m: os.path.join(
    "../data/spin_current_drude/commensurate",
    f"drude_1D_XXZ_M_{M:d}_J_1.0_d_{l}_{m}_T_imag_P_odd_Sz_cons_true_spin_op_basis.txt",
)

files1 = [
    f
    for f in os.listdir("../data/spin_current_drude/other")
    if "drude" in f and "M_8" in f
]
files2 = [
    f
    for f in os.listdir("../data/spin_current_drude/commensurate")
    if "drude" in f and "M_8" in f
]


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
legend_fontsize = 8
label_fontsize = 13
text_fontsize = 10
tick_fontsize = 14

fig, axs = plt.subplots(1, 2, figsize=(6.2, 3))
ax = axs[0]

# ========================= (a) drude vs delta for diffefrent M =========================

deltas1 = []
filenames1 = []
for fname in files1:
    m = re.search(r"d_([0-9.]+)_T", fname)
    if m:
        deltas1.append(float(m.group(1)))
        filenames1.append(path_to_drude1(8, float(m.group(1))))

# Extract two integers next to d_ from files2
lm_pairs2 = []
filenames2 = []

all_deltas = np.hstack([deltas1, [np.abs(np.cos(np.pi * l / m)) for l, m in lm_pairs2]])
all_filenames = np.hstack([filenames1, filenames2])
idx = np.argsort(all_deltas)
all_deltas = np.array(all_deltas)[idx]
all_filenames = np.array(all_filenames)[idx]

Ms = [4, 5, 6, 7, 8, 9]
colors = viridis(np.linspace(0, 1, len(Ms)))
vals = np.zeros((len(Ms), len(all_deltas), 3))

for i, M in enumerate(Ms):
    for j, fname in enumerate(all_filenames):
        new_filename = fname.replace("M_8", f"M_{M}")
        try:
            data = np.abs(np.loadtxt(new_filename)) ** 2
            data = np.sort(data)
            if isinstance(data, float):
                vals[i, j, 0] = data
                vals[i, j, 1] = np.nan
                vals[i, j, 2] = np.nan
            else:
                if i <= 1:
                    vals[i, j, 0] = data[-1]
                    vals[i, j, 1] = data[-2]
                    vals[i, j, 2] = np.nan
                else:
                    vals[i, j, 0] = data[-1]
                    vals[i, j, 1] = data[-2]
                    vals[i, j, 2] = data[-3]
        except FileNotFoundError:
            print(f"File not found: {new_filename}")
            vals[i, j, 0] = np.nan
            vals[i, j, 1] = np.nan
            vals[i, j, 2] = np.nan
    if M == 9:
        # For M==9, linearly interpolate over all_deltas to fill NaNs in vals[i, :, :]
        for k in range(vals.shape[2]):
            y = vals[i, :, k]
            valid = ~np.isnan(y)
            if valid.sum() > 1:
                vals[i, :, k] = np.interp(all_deltas, all_deltas[valid], y[valid])

    if M <= 9:
        ax.plot(
            all_deltas,
            vals[i, :, 0],
            label=rf"$M = {M}$",
            linewidth=2.0,
            linestyle="-",
            color=colors[i],
        )


ax.set_xlabel(r"Anisotropy $\Delta$", fontsize=label_fontsize)
ax.set_ylabel(r"Projection $V^2_{1,1}$", fontsize=label_fontsize)

ax.set_xlim(-0.01, 1.0)
ax.set_ylim(0.0, 1.02)
ax.text(0.06, 0.85, r"\textbf{(a)}", fontsize=text_fontsize, transform=ax.transAxes)

ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.minorticks_on()


# #========================== (b) extrapolated drude =========================

ax2 = axs[1]
mmmax = 201
comm_deltas = [
    np.abs(np.cos(np.pi * l / m))
    for l in np.arange(1, mmmax, 1)
    for m in np.arange(2, mmmax, 1)
]

Dk = lambda l, m: (m / (2 * (m - 1))) * np.sin(np.pi * l / m) ** 2
Dz = lambda l, m: (np.sin(np.pi * l / m) / np.sin(np.pi / m)) ** 2 * (
    1 - m / (2 * np.pi) * np.sin(2 * np.pi / m)
)
Dk_vals = [Dk(l, m) for l in np.arange(1, mmmax, 1) for m in np.arange(2, mmmax, 1)]
Dz_vals = [Dz(l, m) for l in np.arange(1, mmmax, 1) for m in np.arange(2, mmmax, 1)]
comm_idx = np.argsort(comm_deltas)
comm_deltas = np.array(comm_deltas)[comm_idx]
Dk_vals = np.array(Dk_vals)[comm_idx]
Dz_vals = np.array(Dz_vals)[comm_idx]
ax2.plot(
    comm_deltas,
    Dk_vals,
    linestyle="-",
    linewidth=0.5,
    color=colors[0],
    label=r"analytical bound $D_Z$",
)
ax2.plot(
    comm_deltas,
    Dz_vals,
    linestyle="-",
    linewidth=0.5,
    color=colors[2],
    label=r"analytical bound $D_K$",
)
# Extrapolate vals as a linear function of 1/M for each delta
extrapolated_vals = np.zeros((len(all_deltas), 3))
for j in range(len(all_deltas)):
    M_to_fit = np.array([5, 6, 7, 8, 9])
    M_idx = np.array([np.where(Ms == m)[0][0] for m in M_to_fit])
    inv_Ms = 1 / np.array(Ms)[M_idx]
    y = vals[M_idx, j, 0]
    # Linear fit: y = a * (1/M) + b
    coeffs = np.polyfit(inv_Ms, y, 1)
    extrapolated_vals[j, 0] = coeffs[1]


extrapolated_vals = np.array(extrapolated_vals)
neg_idx = extrapolated_vals < 0
extrapolated_vals[neg_idx] = 0.0

idx1 = all_deltas < 0.4
idx2 = all_deltas > 0.4
ax.plot(
    all_deltas[idx1],
    extrapolated_vals[idx1, 0],
    linestyle="--",
    linewidth=2.0,
    color="black",
)
ax.plot(
    all_deltas[idx2],
    extrapolated_vals[idx2, 0],
    linestyle="-",
    linewidth=2.0,
    color="black",
    label=r"$\lim_{M \to \infty} V_{1,1}^2$",
)
ax.legend(
    loc="upper right", fontsize=legend_fontsize, frameon=False, handlelength=1, ncol=2
)
handles = ax.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1]
# Separate into two lists
handles_col1 = handles[:3]
labels_col1 = labels[:3]

handles_col2 = handles[3:]
labels_col2 = labels[3:]


# Create two separate legends
leg1 = ax.legend(
    handles_col1,
    labels_col1,
    loc="upper right",
    bbox_to_anchor=(0.55, 0.98),
    borderaxespad=0.0,
    handlelength=1,
    frameon=False,
    fontsize = legend_fontsize,
)
leg2 = ax.legend(
    handles_col2,
    labels_col2,
    loc="upper right",
    bbox_to_anchor=(1, 0.98),
    borderaxespad=0.0,
    handlelength=1,
    frameon=False,
    fontsize = legend_fontsize,
)

# Add both legends
ax.add_artist(leg1)

ax2.plot(
    all_deltas[idx1],
    extrapolated_vals[idx1, 0],
    linestyle="--",
    linewidth=2.0,
    color=colors[4],
    label=r"$\lim_{M \to \infty} V_{1,1}^2$",
)
ax2.plot(
    all_deltas[idx2],
    extrapolated_vals[idx2, 0],
    linestyle="-",
    linewidth=2.0,
    color=colors[4],
)

ax2.set_xlabel(r"Anisotropy $\Delta$", fontsize=label_fontsize)
ax2.set_xlim(-0.01, 1.0)
ax2.set_ylim(0.0, 1.02)
ax2.text(0.06, 0.85, r"\textbf{(b)}", fontsize=text_fontsize, transform=ax2.transAxes)
ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.minorticks_on()
ax2.legend(loc="best", fontsize=legend_fontsize+2, frameon=False, handlelength=1)
plt.tight_layout()
plt.savefig("figB2.pdf", bbox_inches="tight")
