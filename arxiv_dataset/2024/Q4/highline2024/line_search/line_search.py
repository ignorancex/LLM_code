import matplotlib.pyplot as plt
import numpy as np
import os

# Define time parameters
t_0 = 0
t_f = 60

# Define dimension
d = 400

# Define parameters
Lambda2 = [0.5, 0.25, 0.125, 0.0625, 0.03125]

# Generate an array of distinct colors
colors = plt.cm.plasma(np.linspace(0, 1, len(Lambda2) + 1))

# Define initial conditions
x_0 = np.random.normal(size=d) / np.sqrt(d)
x_star = np.ones(d) / np.sqrt(d)


def generate_orthogonal_matrix(d):
    # Generate a random matrix
    A = np.random.rand(d, d)
    # Compute the QR decomposition
    Q, _ = np.linalg.qr(A)
    return Q


U = generate_orthogonal_matrix(d)
v = U.T @ (x_0 - x_star)
v_0 = np.array([np.sum(v[:d//2]**2), np.sum(v[d//2:]**2)])


def calculate_gamma_limit(lambda1, lambda2):

    numerator = lambda1**3 + 2 * lambda1**2 * \
        lambda2 + lambda2**3 + 2 * lambda2**2 * lambda1
    sqrt_expression = np.sqrt(lambda1**6 - 4 * lambda1**5 * lambda2 + 8 * lambda1**4 * lambda2**2 -
                              6 * lambda1**3 * lambda2**3 + lambda2**6 - 4 * lambda2**5 * lambda1 + 8 * lambda2**4 * lambda1**2)
    numerator -= sqrt_expression
    denominator = (lambda1**2 + lambda2**2)**2

    return numerator / denominator


def calculate_rho(lambda1, lambda2):

    numerator = lambda1**3 - 2 * lambda1**2 * \
        lambda2 - lambda2**3 + 2 * lambda2**2 * lambda1
    sqrt_expression = np.sqrt(lambda1**6 - 4 * lambda1**5 * lambda2 + 8 * lambda1**4 * lambda2**2 -
                              6 * lambda1**3 * lambda2**3 + lambda2**6 - 4 * lambda2**5 * lambda1 + 8 * lambda2**4 * lambda1**2)
    numerator += sqrt_expression
    denominator = 2 * lambda1 * lambda2**2

    return numerator / denominator


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
ax2 = fig2.add_axes([0.2, 0.2, 0.7, 0.7])
fig3 = plt.figure(figsize=(8, 6))
# [left, bottom, width, height] in figure fraction
ax3 = fig3.add_axes([0.2, 0.2, 0.7, 0.7])

for i in range(len(Lambda2)):

    lambda1 = 1
    lambda2 = Lambda2[i]

    gamma_limit = calculate_gamma_limit(lambda1, lambda2)
    rho = calculate_rho(lambda1, lambda2)

    diagonal_entries = np.full(d, lambda2)
    diagonal_entries[:d//2] = lambda1
    D = np.diag(diagonal_entries)
    K = U @ D @ U.T

    def forward_euler(system):

        h = 0.1
        T = int((t_f - t_0) / h)

        t_values = np.linspace(t_0, t_f, T + 1)

        v_values = np.zeros((2, T + 1))
        v_values[:, 0] = v_0

        gamma_values = np.zeros(T + 1)
        gamma_values[0] = gamma_euler(v_0)

        loss_values = np.zeros(T + 1)
        loss_values[0] = loss_euler(v_0)

        for i in range(T):

            v_values[:, i + 1] = v_values[:, i] + h * system(v_values[:, i])

            gamma_values[i+1] = gamma_euler(v_values[:, i+1])
            loss_values[i+1] = loss_euler(v_values[:, i+1])

        return v_values, gamma_values, loss_values, t_values

    def streaming_sgd():

        frequency_to_save = 100
        t_values = np.linspace(t_0, t_f, t_f * d // frequency_to_save + 1)

        v_values = [[] for _ in range(2)]
        v_values[0].append(v_0[0])
        v_values[1].append(v_0[1])

        gamma_values = []
        gamma_i = gamma_sgd(x_0)
        gamma_values.append(gamma_i)

        x = x_0.copy()

        loss_values = []
        loss_i = loss_sgd(x)
        loss_values.append(loss_i)

        for i in range(t_f*d):

            a = U @ np.diag(np.sqrt(diagonal_entries)
                            ) @ np.random.normal(size=d)

            x = x - 1/d * gamma_i * a @ (x - x_star) * a.T

            gamma_i = gamma_sgd(x)
            loss_i = loss_sgd(x)

            if (i+1) % frequency_to_save == 0:

                v = U.T @ (x - x_star)
                v_k = np.array([np.sum(v[:d//2]**2), np.sum(v[d//2:]**2)])
                v_values[0].append(v_k[0])
                v_values[1].append(v_k[1])

                gamma_values.append(gamma_i)
                loss_values.append(loss_i)

        return v_values, gamma_values, loss_values, t_values

    # Define the formula for the stepsize
    def gamma_euler(v):
        return (lambda1**2 * v[0] + lambda2**2 * v[1]) / (lambda1 * v[0] + lambda2 * v[1]) * 2.0 / (lambda1**2 + lambda2**2)

    def gamma_sgd(x):
        return (np.linalg.norm(K @ (x-x_star))**2) / ((x-x_star).T @ K @ (x-x_star)) * 2.0 / (lambda1**2 + lambda2**2)

    # Define the formula for the loss
    def loss_euler(v):
        return 0.5 * (lambda1 * v[0] + lambda2 * v[1])

    def loss_sgd(x):
        return 0.5 * (x-x_star).T @ K @ (x-x_star)

    # Define your system of ODEs here
    def system(v):
        dv1 = -2*gamma_euler(v)*lambda1 * \
            v[0] + gamma_euler(v)**2 * loss_euler(v) * lambda1
        dv2 = -2*gamma_euler(v)*lambda2 * \
            v[1] + gamma_euler(v)**2 * loss_euler(v) * lambda2
        return np.array([dv1, dv2])

    # Solve the system using Forward Euler method
    v_values_euler, gamma_values_euler, loss_values_euler, t_values_euler = forward_euler(
        system)

    # Run streaming SGD to check solutions
    v_values_sgd, gamma_values_sgd, loss_values_sgd, t_values_sgd = streaming_sgd()

    # Plot the solutions
    ax1.plot(t_values_euler, loss_values_euler,
             color=colors[i], linestyle='-', label=r'$\lambda_2 = 1 / 2^{{{}}} $'.format(i+1), linewidth=3)
    ax1.plot(t_values_sgd, loss_values_sgd,
             color=colors[i], linestyle='--', linewidth=3)

    ax2.plot(t_values_euler, v_values_euler[1] / v_values_euler[0],
             color=colors[i], linestyle='-', linewidth=3)
    ax2.plot(t_values_sgd, np.array(v_values_sgd[1]) / np.array(
        v_values_sgd[0]), color=colors[i], linestyle='--', linewidth=3)
    ax2.axhline(y=rho, color=colors[i], linestyle=':', linewidth=3)

    ax3.plot(t_values_euler, gamma_values_euler,
             color=colors[i], linestyle='-', linewidth=3)
    ax3.plot(t_values_sgd, gamma_values_sgd,
             color=colors[i], linestyle='--', linewidth=3)
    ax3.axhline(y=gamma_limit, color=colors[i], linestyle=':', linewidth=3)


ax1.set_xlabel('time (t)')
ax1.set_ylabel(r'risk $\mathscr{R}(t)$')
ax1.tick_params(axis='both', which='major')
ax1.set_yscale('log')
ax1.set_ylim(bottom=1e-7)
ax1.grid()

# Get the handles and labels of the legend
handles, labels = ax1.get_legend_handles_labels()

# Add something to the legend
sgd_handle = plt.Line2D([0], [0], linestyle='--', color='black')
theory_handle = plt.Line2D([0], [0], linestyle='-', color='black')
limit_handle = plt.Line2D([0], [0], linestyle=':', color='black')

# Update the legend
pos = ax1.get_position()
ax1.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
ax1.legend(handles=[sgd_handle, theory_handle, limit_handle] + handles, labels=[
           'SGD', 'theory', 'limit'] + labels, loc='center right', bbox_to_anchor=(1.7, 0.4))

ax2.set_xlabel('time (t)')
ax2.set_ylabel(
    r'quotient $\mathscr{D}_{\lambda_2}(t) / \mathscr{D}_{\lambda_1}(t)$')
ax2.set_yscale('log')
ax2.grid()

ax3.set_xlabel('time (t)')
ax3.set_ylabel(r'learning rate $\gamma_t$')
ax3.set_yscale('log')
ax3.grid()

# Create output directory if it doesn't exist
output_dir = 'figures/line_search_noiseless'
os.makedirs(output_dir, exist_ok=True)

# Save each figure as a PDF
fig1.savefig(os.path.join(output_dir, 'risk.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'quotient.pdf'), bbox_inches='tight')
fig3.savefig(os.path.join(output_dir, 'learning_rate.pdf'), bbox_inches='tight')

# Close the figures to free up memory
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
