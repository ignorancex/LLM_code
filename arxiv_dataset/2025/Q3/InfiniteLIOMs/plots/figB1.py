import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.cm import viridis
import matplotlib.ticker as ticker

filename_spin_op = (
    lambda M, T, P, d, J=1.0: f"../data/1D_XXZ/eigenvalues_1D_XXZ_M_{M:d}_J_{J:.1f}_d_{d}_T_{T}_P_{P}_Sz_cons_true_spin_op_basis.txt"
)
filename_evec = (
    lambda M, T, P, d, J=1.0: f"../data/1D_XXZ/eigenvectors_1D_XXZ_M_{M:d}_J_{J:.1f}_d_{d}_T_{T}_P_{P}_Sz_cons_true_spin_op_basis.txt"
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
legend_fontsize = 10
label_fontsize = 13
text_fontsize = 10
tick_fontsize = 14
points_to_extrapolate = 4

fig, axs = plt.subplots(2, 2, figsize=(6.2, 6))

# ========================= (a) lambdas vs 1/M =========================

ax = axs[0, 0]
Ms = [3, 4, 5, 6, 7, 8, 9]

lambdas = np.zeros(((len(Ms)), 3))

for i, M in enumerate(Ms):
    data = np.loadtxt(filename_spin_op(M, "imag", "odd", "0.75"))
    if i == 0:
        lambdas[i, 0] = data[0]
        lambdas[i, 1] = data[1]
        lambdas[i, 2] = np.nan
    else:
        lambdas[i, 0] = data[0]
        lambdas[i, 1] = data[1]
        lambdas[i, 2] = data[2]

      
Ms = np.array(Ms)

colors = viridis(np.linspace(0, 1, 2 * lambdas.shape[1]))
markers = ["o", "s", "^", "^", "v", "<", ">", "p", "*"]

for col in range(lambdas.shape[1]):
    label = rf"$\lambda_{col + 1}$"

    valid_indices = np.zeros(len(lambdas[:, col]), dtype=bool)
    for k in range(points_to_extrapolate):
      kk = k + 1
      valid_indices[-kk] = True
    log_lambda = np.log(lambdas[valid_indices, col])
    log_Ms_lambda = np.log(1 / Ms[valid_indices])

    # Perform linear regression
    coeffs = np.polyfit(log_Ms_lambda, log_lambda, 1)
    fit_line = np.poly1d(coeffs)

    # Plot the fitted line for the current column
    xx = np.linspace(0.08, 0.6, 100)
    ax.plot(
      xx,
      np.exp(fit_line(np.log(xx))),
      linewidth=1.0,
      linestyle="-",
      color=colors[2*col],
    )
    ax.plot(
      1 / Ms,
      lambdas[:, col],
      label=rf"$\alpha = {col + 1}$",
      # label=rf"$\lambda_{col + 1}\propto M^{{{-coeffs[0]:.2f}}}$",
      linewidth=0.5,
      linestyle=':',
      color=colors[2*col],
      marker=markers[col],
    )

ax.legend(loc="upper left", fontsize=legend_fontsize, frameon=False, handlelength=1.5)
ax.set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)

ax.set_xlim(0.09, 0.4)
ax.set_ylim(1e-4, 1)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_xticks(
    [1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10],
    [
        r"$\frac{1}{3}$",
        r"$\frac{1}{4}$",
        r"$\frac{1}{5}$",
        r"$\frac{1}{6}$",
        r"$\frac{1}{7}$",
        r"$\frac{1}{8}$",
        r"$\frac{1}{9}$",
        r"$\frac{1}{10}$",
        # r"$0$",
    ],
    fontsize=tick_fontsize,
)

ax.text(0.06, 0.2, r"\textbf{(a)} $\Delta=\frac{3}{4}$", fontsize=text_fontsize, transform=ax.transAxes)


#========================== (c) projection on spin current =========================

Ms = [3, 4, 5, 6, 7, 8, 9]
proj = np.zeros(((len(Ms)), 3))
ax = axs[1, 0]

for i, M in enumerate(Ms):
    data = np.loadtxt(filename_evec(M, "imag", "odd", "0.75"))
    if i == 0:
        proj[i, 0] = data[0, 0]
        proj[i, 1] = data[0, 1]
        proj[i, 2] = np.nan
    else:
        proj[i, 0] = data[0, 0]
        proj[i, 1] = data[0, 1]
        proj[i, 2] = data[0, 2]

      
Ms = np.array(Ms)

colors = viridis(np.linspace(0, 1, 2 * lambdas.shape[1]))
markers = ["o", "s", "^", "^", "v", "<", ">", "p", "*"]

for col in range(proj.shape[1]):
    label = rf"$\alpha = {col + 1}$"

    valid_indices = np.zeros(len(proj[:, col]), dtype=bool)
    for k in range(points_to_extrapolate):
      kk = k + 1
      valid_indices[-kk] = True

    # # Perform linear regression
    coeffs = np.polyfit(1/Ms[valid_indices], np.abs(proj[valid_indices, col])**2, 1)
    fit_line = np.poly1d(coeffs)

    # # Plot the fitted line for the current column
    xx = np.linspace(0.00, 0.6, 100)
    ax.plot(
      xx,
      fit_line(xx),
      linewidth=1.0,
      linestyle="-",
      color=colors[2*col],
    )
    ax.plot(
      1 / Ms,
      np.abs(proj[:, col])**2,
      # label=rf"$\lambda_{col + 1}\propto M^{{{-coeffs[0]:.2f}}}$",
      label = label,
      linewidth=0.5,
      linestyle=':',
      color=colors[2*col],
      marker=markers[col],
    )

ax.legend(loc="upper left", fontsize=legend_fontsize, frameon=False, handlelength=1.5)
ax.set_xlabel(r"$1/M$", fontsize=label_fontsize)
ax.set_ylabel(r"Projection $V^2_{\alpha,1}$", fontsize=label_fontsize)

ax.set_xlim(0.0, 0.4)
ax.set_ylim(-0.05,1.0)
ax.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_xticks(
    [1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10, 0],
    [
        r"$\frac{1}{3}$",
        r"$\frac{1}{4}$",
        r"$\frac{1}{5}$",
        r"$\frac{1}{6}$",
        # r"$\frac{1}{7}$",
        "",
        r"$\frac{1}{8}$",
        # r"$\frac{1}{9}$",
        "",
        r"$\frac{1}{10}$",
        r"$0$",
    ],
    fontsize=tick_fontsize,
)
ax.text(0.06, 0.6, r"\textbf{(b)} $\Delta=\frac{3}{4}$", fontsize=text_fontsize, transform=ax.transAxes)


# ========================= (b) lambdas vs 1/M =========================

ax = axs[0, 1]
Ms = [3, 4, 5, 6, 7, 8, 9]

lambdas = np.zeros(((len(Ms)), 3))

for i, M in enumerate(Ms):
    data = np.loadtxt(filename_spin_op(M, "imag", "odd", "0.1"))
    if i == 0:
        lambdas[i, 0] = data[0]
        lambdas[i, 1] = data[1]
        lambdas[i, 2] = np.nan
    else:
        lambdas[i, 0] = data[0]
        lambdas[i, 1] = data[1]
        lambdas[i, 2] = data[2]

      
Ms = np.array(Ms)

# Generate colors from the viridis palette
colors = viridis(np.linspace(0, 1, 2 * lambdas.shape[1]))
# colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "cyan"]
markers = ["o", "s", "^", "^", "v", "<", ">", "p", "*"]

for col in range(lambdas.shape[1]):
    label = rf"$\lambda_{col + 1}$"

    # Fit a line to the log-log data for the current column (excluding NaNs)
    # valid_indices = ~np.isnan(lambdas[:, col])  # Exclude the first point and NaNs
    valid_indices = np.zeros(len(lambdas[:, col]), dtype=bool)
    for k in range(points_to_extrapolate):
      kk = k + 1
      valid_indices[-kk] = True
    log_lambda = np.log(lambdas[valid_indices, col])
    log_Ms_lambda = np.log(1 / Ms[valid_indices])

    # Perform linear regression
    coeffs = np.polyfit(log_Ms_lambda, log_lambda, 1)
    fit_line = np.poly1d(coeffs)

    # Plot the fitted line for the current column
    xx = np.linspace(0.08, 0.6, 100)
    ax.plot(
      xx,
      np.exp(fit_line(np.log(xx))),
      linewidth=1.0,
      linestyle="-",
      color=colors[2*col],
    )
    ax.plot(
      1 / Ms,
      lambdas[:, col],
      label=rf"$\alpha = {col + 1}$",
      # label=rf"$\lambda_{col + 1}\propto M^{{{-coeffs[0]:.2f}}}$",
      linewidth=0.5,
      linestyle=':',
      color=colors[2*col],
      marker=markers[col],
    )

ax.legend(loc="upper left", fontsize=legend_fontsize, frameon=False, handlelength=1.5)
# ax.set_xlabel(r"$1/M$", fontsize=label_fontsize)
# ax.set_ylabel(r"Eigenvalue $\lambda_{\alpha}$", fontsize=label_fontsize)

ax.set_xlim(0.09, 0.4)
ax.set_ylim(8*1e-5, 1)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_xticks(
    [1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10],
    [
        r"$\frac{1}{3}$",
        r"$\frac{1}{4}$",
        r"$\frac{1}{5}$",
        r"$\frac{1}{6}$",
        r"$\frac{1}{7}$",
        r"$\frac{1}{8}$",
        r"$\frac{1}{9}$",
        r"$\frac{1}{10}$",
        # r"$0$",
    ],
    fontsize=tick_fontsize,
)
ax.set_yticklabels([])
ax.text(0.06, 0.6, r"\textbf{(c)} $\Delta=0.10$", fontsize=text_fontsize, transform=ax.transAxes)


#========================== (c) projection on spin current =========================

Ms = [3, 4, 5, 6, 7, 8, 9]
proj = np.zeros(((len(Ms)), 3))
ax = axs[1, 1]

for i, M in enumerate(Ms):
    data = np.loadtxt(filename_evec(M, "imag", "odd", "0.1"))
    if i == 0:
        proj[i, 0] = data[0, 0]
        proj[i, 1] = data[0, 1]
        proj[i, 2] = np.nan
    else:
        proj[i, 0] = data[0, 0]
        proj[i, 1] = data[0, 1]
        proj[i, 2] = data[0, 2]

      
Ms = np.array(Ms)

# Generate colors from the viridis palette
colors = viridis(np.linspace(0, 1, 2 * lambdas.shape[1]))
# colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "cyan"]
markers = ["o", "s", "^", "^", "v", "<", ">", "p", "*"]

for col in range(proj.shape[1]):
    label = rf"$\alpha = {col + 1}$"

    ax.plot(
      1 / Ms,
      np.abs(proj[:, col])**2,
      # label=rf"$\lambda_{col + 1}\propto M^{{{-coeffs[0]:.2f}}}$",
      label = label,
      linewidth=0.5,
      linestyle=':',
      color=colors[2*col],
      marker=markers[col],
    )

ax.legend(loc="best", fontsize=legend_fontsize, frameon=False, handlelength=1.5)
ax.set_xlabel(r"$1/M$", fontsize=label_fontsize)

ax.set_xlim(0.0, 0.4)
ax.set_ylim(-0.05,1.0)
ax.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_xticks(
    [1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10, 0],
    [
        r"$\frac{1}{3}$",
        r"$\frac{1}{4}$",
        r"$\frac{1}{5}$",
        r"$\frac{1}{6}$",
        # r"$\frac{1}{7}$",
        "",
        r"$\frac{1}{8}$",
        # r"$\frac{1}{9}$",
        "",
        r"$\frac{1}{10}$",
        r"$0$",
    ],
    fontsize=tick_fontsize,
)
ax.text(0.6, 0.6, r"\textbf{(d)} $\Delta=0.10$", fontsize=text_fontsize, transform=ax.transAxes)
ax.set_yticklabels([])


plt.tight_layout()
plt.savefig("figB1.pdf", bbox_inches="tight")
