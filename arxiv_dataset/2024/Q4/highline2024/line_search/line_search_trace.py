import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# Define time parameters
t_0 = 0
t_f = 60

# Define dimension
d = 400

# Define parameters
S = [3.5, 4.0, 5.5, 7.5]

# Generate an array of distinct colors
colors = plt.cm.plasma(np.linspace(0, 1, len(S) + 1))

# Define initial conditions
x_0 = np.random.normal(size=d) / np.sqrt(d)
x_star = np.ones(d) / np.sqrt(d)

v = x_0 - x_star
v_0 = v**2

# Activate LaTeX support in Matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathrsfs}",
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    'font.size': 30,               # Default font size for text elements
    'axes.titlesize': 30,          # Font size for axis titles
    'axes.labelsize': 30,          # Font size for axis labels
    'xtick.labelsize': 30,         # Font size for x-axis tick labels
    'ytick.labelsize': 30,         # Font size for y-axis tick labels
})

# Plotting
fig1 = plt.figure(figsize=(13, 6))
# [left, bottom, width, height] in figure fraction
ax1 = fig1.add_axes([0.125, 0.2, 0.6, 0.7])
fig2 = plt.figure(figsize=(8, 6))
# [left, bottom, width, height] in figure fraction
ax2 = fig2.add_axes([0.27, 0.2, 0.7, 0.7])

for i in range(len(S)):

    s = S[i]

    SigmaPsi = np.array([np.power(float(i) / (d+1), -1 / s)
                        for i in range(1, d+1)])
    SigmaPsi *= np.sqrt((d / np.sum(SigmaPsi*SigmaPsi)))

    diagonal_entries = SigmaPsi**2
    lambdaMin = np.min(diagonal_entries)
    NormalizedTraceKSquared = np.sum(diagonal_entries**2) / d

    K = np.diag(diagonal_entries)

    def heun():

        h = 5e-5
        T = int((t_f - t_0) / h)

        t_values = np.linspace(t_0, t_f, T + 1)

        v_values_lineSearch = np.zeros((d, T + 1))
        v_values_lineSearch[:, 0] = v_0

        gamma_values_lineSearch = np.zeros(T + 1)
        gamma_values_lineSearch[0] = gamma_heun(v_0)

        loss_values_lineSearch = np.zeros(T + 1)
        loss_values_lineSearch[0] = loss_heun(v_0)

        v_values_Polyak = np.zeros((d, T + 1))
        v_values_Polyak[:, 0] = v_0

        loss_values_Polyak = np.zeros(T + 1)
        loss_values_Polyak[0] = loss_heun(v_0)

        for i in range(T):

            v_tilde_lineSearch = v_values_lineSearch[:, i] + \
                h * system_lineSearch(v_values_lineSearch[:, i])
            v_values_lineSearch[:, i + 1] = v_values_lineSearch[:, i] + h / 2 * (
                system_lineSearch(v_tilde_lineSearch) + system_lineSearch(v_values_lineSearch[:, i]))

            gamma_values_lineSearch[i +
                                    1] = gamma_heun(v_values_lineSearch[:, i+1])
            loss_values_lineSearch[i +
                                   1] = loss_heun(v_values_lineSearch[:, i+1])

            v_tilde_Polyak = v_values_Polyak[:, i] + \
                h * system_Polyak(v_values_Polyak[:, i])
            v_values_Polyak[:, i + 1] = v_values_Polyak[:, i] + h / 2 * \
                (system_Polyak(v_tilde_Polyak) +
                 system_Polyak(v_values_Polyak[:, i]))

            loss_values_Polyak[i+1] = loss_heun(v_values_Polyak[:, i+1])

        return gamma_values_lineSearch, loss_values_lineSearch, loss_values_Polyak, t_values

    def streaming_sgd():

        frequency_to_save = 1000
        t_values = np.linspace(t_0, t_f, t_f * d // frequency_to_save + 1)

        gamma_values_lineSearch = []
        gamma_i_lineSearch = gamma_sgd(x_0)
        gamma_values_lineSearch.append(gamma_i_lineSearch)

        x_lineSearch = x_0.copy()

        loss_values_lineSearch = []
        loss_i_lineSearch = loss_sgd(x_lineSearch)
        loss_values_lineSearch.append(loss_i_lineSearch)

        x_Polyak = x_0.copy()

        loss_values_Polyak = []
        loss_i_Polyak = loss_sgd(x_Polyak)
        loss_values_Polyak.append(loss_i_Polyak)

        for i in range(t_f*d):

            a = np.diag(np.sqrt(diagonal_entries)) @ np.random.normal(size=d)

            x_lineSearch = x_lineSearch - 1/d * \
                gamma_i_lineSearch * a @ (x_lineSearch - x_star) * a.T

            gamma_i_lineSearch = gamma_sgd(x_lineSearch)
            loss_i_lineSearch = loss_sgd(x_lineSearch)

            x_Polyak = x_Polyak - 1/d * a @ (x_Polyak - x_star) * a.T

            loss_i_Polyak = loss_sgd(x_Polyak)

            if (i+1) % frequency_to_save == 0:

                gamma_values_lineSearch.append(gamma_i_lineSearch)
                loss_values_lineSearch.append(loss_i_lineSearch)
                loss_values_Polyak.append(loss_i_Polyak)

        return gamma_values_lineSearch, loss_values_lineSearch, loss_values_Polyak, t_values

    # Define the formula for the stepsize
    def gamma_heun(v):
        return (diagonal_entries**2 @ v) / (diagonal_entries @ v * NormalizedTraceKSquared)

    def gamma_sgd(x):
        return (np.linalg.norm(K @ (x-x_star))**2) / ((x-x_star).T @ K @ (x-x_star) * NormalizedTraceKSquared)

    # Define the formula for the loss
    def loss_heun(v):
        return 0.5 * (diagonal_entries @ v)

    def loss_sgd(x):
        return 0.5 * (x-x_star).T @ K @ (x-x_star)

    # Define your system of ODEs here
    def system_lineSearch(v):
        return -2 * gamma_heun(v) * v * diagonal_entries + 2 / d * gamma_heun(v)**2 * loss_heun(v) * diagonal_entries

    def system_Polyak(v):
        return -2 * v * diagonal_entries + 2 / d * loss_heun(v) * diagonal_entries

    # Solve the system using Forward Euler method
    gamma_values_lineSearch_heun, loss_values_lineSearch_heun, loss_values_Polyak_heun, t_values_heun = heun()

    # Run streaming SGD to check solutions
    gamma_values_lineSearch_sgd, loss_values_lineSearch_sgd, loss_values_Polyak_sgd, t_values_sgd = streaming_sgd()

    # Plot the solutions
    ax1.plot(t_values_heun, loss_values_lineSearch_heun,
             color=colors[i], linestyle='-', label=f'$\\frac{1}{{d}} Tr(K^2) \\approx {np.round(NormalizedTraceKSquared, decimals=2)}$', linewidth=3)
    ax1.plot(t_values_sgd, loss_values_lineSearch_sgd,
             color=colors[i], linestyle='--', linewidth=3)
    ax1.plot(t_values_heun, loss_values_Polyak_heun, color=colors[i], linestyle='-', linewidth=3, marker='o',
             markersize=7, markevery=np.linspace(0, len(t_values_heun)-1, num=12, endpoint=True).astype(int))
    ax1.plot(t_values_sgd, loss_values_Polyak_sgd, color=colors[i], linestyle='--', linewidth=3, marker='o',
             markersize=7, markevery=np.linspace(0, len(t_values_sgd)-1, num=12, endpoint=True).astype(int))

    ax2.plot(t_values_heun, (gamma_values_lineSearch_heun * NormalizedTraceKSquared) /
             lambdaMin, color=colors[i], linestyle='-', linewidth=3)
    ax2.plot(t_values_sgd, (np.array(gamma_values_lineSearch_sgd) *
             NormalizedTraceKSquared) / lambdaMin, color=colors[i], linestyle='--', linewidth=3)
    ax2.plot(t_values_sgd, np.zeros(len(t_values_sgd)) + NormalizedTraceKSquared / lambdaMin,
             color=colors[i], linestyle='-', linewidth=3, marker='o', markersize=7, markevery=np.linspace(0, len(t_values_sgd)-1, num=12, endpoint=True).astype(int))


ax1.set_xlabel('time (t)')
ax1.set_ylabel(r'risk $\mathscr{R}(t)$')
ax1.set_yscale('log')
ax1.grid()

# Get the handles and labels of the legend
handles, labels = ax1.get_legend_handles_labels()

# Add something to the legend
lineSearch_sgd_handle = plt.Line2D(
    [0], [0], linestyle='--', linewidth=3, color='black')
lineSearch_heun_handle = plt.Line2D(
    [0], [0], linestyle='-', linewidth=3, color='black')
Polyak_heun_handle = plt.Line2D(
    [0], [0], linestyle='-', linewidth=3, marker='o', markersize=7, color='black')
Polyak_sgd_handle = plt.Line2D(
    [0], [0], linestyle='--', linewidth=3, marker='o', markersize=7, color='black')

# Update the legend
pos = ax1.get_position()
ax1.set_position([pos.x0, pos.y0, pos.width * 0.745, pos.height])
ax1.legend(handles=[lineSearch_sgd_handle, lineSearch_heun_handle, Polyak_sgd_handle, Polyak_heun_handle] + handles, labels=[
           'SGD (line search)', 'theory (line search)', 'SGD (Polyak)', 'theory (Polyak)'] + labels, loc='upper right', bbox_to_anchor=(1.9, 1.1))

# Fill the area between the lines
y1 = np.ones_like(t_values_sgd) * 1
y2 = np.ones_like(t_values_sgd) * 2
ax2.fill_between(t_values_sgd, y1, y2, color='gray', alpha=0.3)

ax2.set_xlabel('time (t)')
ax2.set_ylabel(
    r'quotient $\gamma_t / \frac{\lambda_{\min}(K)}{\frac{1}{{d}} Tr(K^2)}$')
ax2.set_yscale('log')
ax2.grid()

# Create output directory if it doesn't exist
output_dir = 'figures/line_search_vs_polyak'
os.makedirs(output_dir, exist_ok=True)

# Save each figure as a PDF
fig1.savefig(os.path.join(output_dir, 'trace_risk.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'trace_quotient.pdf'),
             bbox_inches='tight')

# Close the figures to free up memory
plt.close(fig1)
plt.close(fig2)
