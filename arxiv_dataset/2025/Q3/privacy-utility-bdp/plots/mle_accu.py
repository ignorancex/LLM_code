import numpy as np
from math import comb
from math import comb, floor, ceil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import bisect
from functools import lru_cache


# ----------------------
# INPUT
# -----------------
n = 700  # Data Size
markov_probs = [0.2, 0.3, 0.4]  # Markov prob of changing state
beta_target = 0.05  # Beta
theta = 0.5
n1 = int(n * 0.5)  # Proportion of 1s in the true data
# ----------------------
# Derived parameters
# ----------------------
gamma = {}
min_eps = {}
eps_vals = {}
for a in markov_probs:
    gamma[a] = (1 - a) / a
    min_eps[a] = np.ceil(np.log(gamma[a]) * 4)
    print(min_eps[a])
    eps_vals[a] = np.linspace(min_eps[a], 10, 5)
    # eps_vals[a] = np.linspace(10,15,3)


# ------------------------
# Binomial PMF
# ------------------------
@lru_cache
def binom_pmf(k, n, q):
    return comb(n, k) * (q**k) * ((1 - q) ** (n - k))


# PMF Bin(n1, q1) y Bin(n2, q2)
def conv_binom_pmf(k, n1, q1, n2, q2):
    # Z = X + Y, where X ~ Bin(n1, q1), Y ~ Bin(n2, q2)
    return sum(
        binom_pmf(i, n1, q1) * binom_pmf(k - i, n2, q2)
        for i in range(max(0, k - n2), min(n1, k) + 1)
    )


# Uper tail: P(Z >= t)
def conv_binom_tail(t, n1, q1, n2, q2):
    max_k = n1 + n2
    if t > max_k:
        return 0.0
    return sum(conv_binom_pmf(k, n1, q1, n2, q2) for k in range(t, max_k + 1))


# Distribution: P(Z <= t)
def conv_binom_dist(t, n1, q1, n2, q2):
    if t < 0:
        return 0.0
    return sum(conv_binom_pmf(k, n1, q1, n2, q2) for k in range(0, min(t, n1 + n2) + 1))


# ------------------------
# Accuracy of [8]
# ------------------------
# Probability of error alpha
def compute_beta_rr(alpha, q, scale, n):
    n1 = int(n * 0.5)  # Change if theta changes
    t_0 = floor(n1 * scale + n * q - (alpha * scale))
    t_1 = ceil(n1 * scale + n * q + (alpha * scale))
    return conv_binom_tail(t_1, n1, 1 - q, n - n1, q) + conv_binom_dist(
        t_0, n1, 1 - q, n - n1, q
    )


# beta given alpha
def find_alpha_for_beta_rr(beta, q, scale, n, alpha_max=1000, tol=1e-4):
    def f(alpha):
        value = compute_beta_rr(alpha, q, scale, n) - beta
        return value

    print(f(0), f(alpha_max))
    return bisect(f, 0, alpha_max, xtol=tol)


# ------------------------
# Accuracy of BDP Laplace mechanism (ours)
# ------------------------
def find_alpha_for_beta_laplace(beta, scale):
    return -scale * np.log(beta)


# ----------------------
# MAIN
# ----------------------

alphas_rr_vs_eps = {a: [] for a in markov_probs}
alphas_laplace_vs_eps = {a: [] for a in markov_probs}
for a in markov_probs:
    print(f"computing α={a}")
    for eps_val in eps_vals[a]:
        print(f"computing eps={eps_val}")
        # Parameters BDP RR
        rho = (
            4
            + a * (a * np.exp(eps_val) - 2)
            - np.sqrt(a**2 * np.exp(eps_val) * (4 + a * (a * np.exp(eps_val) - 4)))
        ) / (8 + 2 * a * (a * np.exp(eps_val) + a - 4))
        p_rr = 1 - rho

        # Parameters BDP Laplace
        eps_laplace_val = eps_val - 4 * np.log(gamma[a])
        scale_lap_val = 1 / (eps_laplace_val)

        # Accuracy RR for one true proportion theta
        scale_rr = 2 * p_rr - 1
        q = rho
        alpha_rr = find_alpha_for_beta_rr(beta_target, q, scale_rr, n)
        alphas_rr_vs_eps[a].append(alpha_rr)
        # Accuracy Laplace (independent of true proportion)
        alpha_laplace_val = find_alpha_for_beta_laplace(beta_target, scale_lap_val)
        alphas_laplace_vs_eps[a].append(alpha_laplace_val)

# ----------------------
# PLOT 1
# ----------------------

plt.figure(figsize=(9, 5))
plt.yscale('log')

for a in markov_probs:
    # Datos
    x_rr = eps_vals[a]
    y_rr = alphas_rr_vs_eps[a]

    x_lap = eps_vals[a]
    y_lap = alphas_laplace_vs_eps[a]
    # Plot RR
    plt.plot(eps_vals[a], alphas_rr_vs_eps[a], color=plt.cm.Blues(a + 0.4), linewidth=3)
    plt.scatter(
        x_rr,
        y_rr,
        color=plt.cm.Blues(a + 0.4),
        marker="x",
        s=60,
        linewidths=3,
        label=rf"Sota [8] $P_{{ss}}={1-a}$",
    )
    # Plot Laplace
    plt.plot(
        eps_vals[a],
        alphas_laplace_vs_eps[a],
        color=plt.cm.Greens(a + 0.4),
        linestyle="-",
        linewidth=3,
    )
    plt.scatter(
        x_lap,
        y_lap,
        color=plt.cm.Greens(a + 0.4),
        marker="o",
        s=60,
        zorder=3,
        label=rf"Ours $P_{{ss}}={1-a}$",
    )

plt.xlabel(r"$\varepsilon$", fontsize=16)
plt.ylabel(r"$\alpha$", fontsize=16)
plt.title(r"$\alpha$ vs. $\varepsilon$ (fixed $\beta = 0.05$)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=14, loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tick_params(axis="both", labelsize=14)
plt.tight_layout()

# Save and plot
filename = f"accu_counting_beta{beta_target}_n{n}.png"
plt.savefig(filename, dpi=300)
print(f"Plot saved as {filename}")
plt.show()
