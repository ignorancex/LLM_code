import os

from matplotlib import pyplot as plt
import numpy as np

COLORS = ["red", "green", "blue", "orange", "purple", "cyan"]
MARKERS = ['x', '.', '+', '1']
LINES = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10)), (0, (5, 10))]
HATCHES = ['||', '//', '--']


def plot_setup(max_y, min_y=0.0, max_x=1.0, min_x=0.0, x_label='ρ', y_label='BDPL', y_rot='vertical', x_scale='linear',
               y_scale='linear', fontsize=20):
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize, rotation=y_rot)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.yscale(y_scale)
    plt.xscale(x_scale)
    plt.ylim((min_y, max_y))
    plt.xlim((min_x, max_x))


# Return general bound for comparison with gaussian bound
def gaussian_general_bound(n, b_minus_a=1, d_privacy_eps=1):
    bdpls_old = n * b_minus_a * d_privacy_eps
    return [0.0, 1.0], [bdpls_old, bdpls_old]


# Return Gaussian bound
def gaussian_new_bound(n, b_minus_a=1, d_privacy_eps=1):
    max_rho = min(1.0, 1 / (n - 2))
    rhos = np.linspace(0.0, max_rho, 1000)
    bdpls = (((n / 2) ** 2 / ((1 / rhos) - n + 2)) + 1) * b_minus_a * d_privacy_eps

    return rhos, bdpls


# Plot Gaussian bound and general bound
def gaussian_bounds(ns, max_y, b_minus_a=1, d_privacy_eps=1, legend=False, save=False):
    max_rho = 1.0 / (np.amin(np.array(ns)) - 2)
    plot_setup(max_y, max_x=max_rho, x_label="$\\rho$", y_label="BDPL", fontsize=25)

    for i, n in enumerate(ns):
        new_xs, new_ys = gaussian_new_bound(n, b_minus_a, d_privacy_eps)
        plt.plot(new_xs, new_ys, color=COLORS[i], linestyle=LINES[0], linewidth=3, label=f'Gaussian bound; $n = {int(n)}$')
        old_xs, old_ys = gaussian_general_bound(n, b_minus_a, d_privacy_eps)
        plt.plot(old_xs, old_ys, color=COLORS[i], linestyle=LINES[1], linewidth=3, label=f'General bound; $n = {int(n)}$')

    if legend:
        plt.legend(fontsize=25, bbox_to_anchor=(1.05, 1.0))
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/gaussian_bound_ns_{ns}.pdf', bbox_inches='tight')
    plt.show()


def markov_bounds_new(max_y, legend=False, save=False):
    plot_setup(max_y, min_y=1, max_x=5, y_scale='log', x_label='DP leakage', y_label='BDP leakage', fontsize=25)

    gammas = [5, 10, 50, 100]
    n = 100

    epsilons = np.linspace(0, 5, 1000)
    general_bound = epsilons * n
    plt.plot(epsilons, general_bound, color=COLORS[0], linestyle=LINES[0], linewidth=3, label='General bound')

    for i, gamma in enumerate(gammas):
        markov_chain_bound = epsilons + 4 * np.log(gamma)
        plt.plot(epsilons, markov_chain_bound, color=COLORS[i+1], linestyle=LINES[i+1], linewidth=3,
                 label=f'Markov chain bound; γ = {gamma}')

    if legend:
        plt.legend(loc="upper center", bbox_to_anchor=(1.75, 1.0), fontsize=20)
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/markov_bounds_compared.pdf', bbox_inches='tight')
    plt.show()


# Minimum probability beta of a high error for a BDP algorithm for arbitrary data distribution
def arbitrary_utility(save=False):
    plot_setup(1.0, min_y=0.001, max_x=5.0, min_x=0, x_label='ε', y_label='β', y_scale='log', y_rot='horizontal')

    bdp_epsilons = np.linspace(0.0, 5.0, 1000)
    betas = 1.0 / (np.exp(bdp_epsilons) + 1)

    plt.plot(bdp_epsilons, betas, color=COLORS[0], linewidth=3)
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/arbitrary_bound.pdf', bbox_inches='tight')
    plt.show()


# Markov chain bound and general bound compared
def markov_bounds(ns, dp_epsilon):
    plot_setup(max_y=45, max_x=100, min_x=1, x_label="ratio of maximum and minimum transition probability",
               y_label="BDPL", x_scale="log")

    ratios = np.linspace(1.0, 100.0, 1000)
    new_bound = dp_epsilon + 4 * np.log(ratios)
    plt.plot(ratios, new_bound)

    for n in ns:
        plt.plot([1, 100], [n * dp_epsilon, n * dp_epsilon], label=f"general bound for n = {n}")

    plt.legend()
    plt.show()


# Values of n and ratio γ where the new Markov chain bound improves over general bound.
def markov_new_bound_better(dp_epsilons, max_n=100, save=False):
    plot_setup(max_y=10000, min_y=1, max_x=max_n, min_x=0, x_label='$n$', y_label='$\\gamma$', y_scale='log', y_rot='horizontal')

    ns = np.linspace(1, max_n, 1000)
    for i, dp_epsilon in enumerate(dp_epsilons):
        alphas = np.exp(((ns - 1)/4) * dp_epsilon)
        plt.fill_between(ns, alphas, facecolor='none', hatch=HATCHES[i], edgecolor=COLORS[i], linewidth=3, label=f'$\\varepsilon = {dp_epsilon}$')

    plt.legend(fontsize=20, bbox_to_anchor=(1.5, 1.0))
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/markov_bound_improvement_epsilons_{dp_epsilons}.pdf', bbox_inches='tight')
    plt.show()


# Maximum transition probability ratio γ so that Markov chain bound is only h times worse than DP privacy budget
def markov_utility(dp_epsilons, save=False):
    plot_setup(max_y=100, min_y=1, max_x=10, min_x=1, x_label='h', y_label='γ', y_rot='horizontal', y_scale='log')

    hs = np.linspace(1, 10, 1000)
    for i, dp_epsilon in enumerate(dp_epsilons):
        gammas = np.exp(dp_epsilon * (hs - 1)/(4 * hs))
        plt.plot(hs, gammas, linestyle=LINES[i], color=COLORS[i], label=f"ε = {dp_epsilon}")

    plt.legend(fontsize=15, loc='upper right')
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/markov_utility_eps_{dp_epsilons}.pdf', bbox_inches='tight')
    plt.show()

def markov_utility_reverse_axes(dp_epsilons, save=False):
    plot_setup(max_y=10, min_y=1, max_x=100, min_x=1, x_label='$\\gamma$', y_label='$h$', y_rot='horizontal', x_scale='log')

    gammas = np.logspace(0, 2, 1000)
    for i, dp_epsilon in enumerate(dp_epsilons):
        hs = 1. / (1. - (4 * np.log(gammas) / dp_epsilon))
        hs = hs[hs >= 0]
        plt.plot(gammas[:len(hs)], hs, linestyle=LINES[i], color=COLORS[i], linewidth=3, label=f"$\\varepsilon = {dp_epsilon}$")


    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1.0))
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/markov_utility_eps_{dp_epsilons}_reverse_axes.pdf', bbox_inches='tight')
    plt.show()

def gaussian_utility_reverse_axes(ns, save=False):
    plot_setup(max_y=10, min_y=1, max_x=1, min_x=0, x_label='$\\rho$', y_label='$h$', y_rot='horizontal')

    rhos = np.linspace(0, 1, 1000)
    for i, n in enumerate(ns):
        hs = (n ** 2 / (4 * ((1 / rhos) - n + 2))) + 1
        hs = hs[hs >= 0]
        plt.plot(rhos[:len(hs)], hs, linestyle=LINES[i], linewidth=3, color=COLORS[i], label=f'$n = {int(n)}$')

    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1.0))
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/gaussian_utility_reverse_axes.pdf', bbox_inches='tight')
    plt.show()

# alpha, beta accuracy of laplace algorithm for BDP for arbitrary correlation
def arbitrary_laplace(ms, beta, save=False):
    plot_setup(max_y=100, min_y=1, min_x=0, max_x=5, x_label='ε', y_label='α', y_scale='log', y_rot='horizontal')

    epsilons = np.linspace(0, 5, 1000)

    for i, m in enumerate(ms):
        alphas = np.log(1 / beta) * (m / epsilons)
        plt.plot(epsilons, alphas, linestyle=LINES[i], color='k', label=f"m = {m}")

    plt.legend(fontsize=15)
    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/arbitrary_alpha_beta_{beta}_m_{ms}.pdf', bbox_inches='tight')
    plt.show()


# Probability beta of a large error for any BDP algorithm for arbitrary correlation
def arbitrary_prob_error_greater_than_half_dataset(save=False):
    plot_setup(max_y=1, min_y=0, max_x=5, min_x=0, x_label='ε', y_label='β', y_rot='horizontal')
    epsilons = np.linspace(0, 5, 1000)
    betas = np.exp(-epsilons/2)
    plt.plot(epsilons, betas, color=COLORS[0], linewidth=3)

    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/arbitrary_counting_query_large_error.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    SAVE = True

    gaussian_bounds([3, 5, 7], 10, legend=True, save=SAVE)

    markov_new_bound_better(dp_epsilons=[0.5, 1, 2], save=SAVE)

    markov_utility_reverse_axes(dp_epsilons=[0.5, 1, 2, 5, 10, 20], save=SAVE)

    gaussian_utility_reverse_axes(ns=[3, 6, 12, 24], save=SAVE)

    # Utility experiment results are directly plotted in the utility_experiment.py file
