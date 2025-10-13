import numpy as np
from math import comb, floor, ceil
import matplotlib.pyplot as plt
from functools import lru_cache


# ------------------------
# Binomial PMF and convolution
# ------------------------
@lru_cache
def binom_pmf(k, n, q):
    return comb(n, k) * (q**k) * ((1 - q) ** (n - k))


def conv_binom_pmf(k, n1, q1, n2, q2):
    return sum(
        binom_pmf(i, n1, q1) * binom_pmf(k - i, n2, q2)
        for i in range(max(0, k - n2), min(n1, k) + 1)
    )


def conv_binom_tail(t, n1, q1, n2, q2):
    max_k = n1 + n2
    if t > max_k:
        return 0.0
    return sum(conv_binom_pmf(k, n1, q1, n2, q2) for k in range(t, max_k + 1))


def conv_binom_dist(t, n1, q1, n2, q2):
    if t < 0:
        return 0.0
    return sum(conv_binom_pmf(k, n1, q1, n2, q2) for k in range(0, t + 1))


# ------------------------
# Beta computation
# ------------------------
def compute_beta_rr(alpha, q, scale, n):
    n1 = int(n * 0.5)
    t_0 = floor(n1 * scale + n * q - (alpha * scale))
    t_1 = ceil(n1 * scale + n * q + (alpha * scale))
    return conv_binom_tail(t_1, n1, 1 - q, n - n1, q) + conv_binom_dist(
        t_0, n1, 1 - q, n - n1, q
    )


def compute_beta_laplace(alpha, scale):
    return np.exp(-alpha / scale)


# ------------------------
# Parameters
# ------------------------
n = 500
a = 0.2
gamma = (1 - a) / a
theta = 0.5
alphas = np.linspace(1, 7, 7)
epsilons = np.linspace(13, 15, 3)

# ------------------------
# Computation for different epsilons
# ------------------------
results_rr = []
results_laplace = []

for eps in epsilons:
    # RR parameters
    rho = (
        4
        + a * (a * np.exp(eps) - 2)
        - np.sqrt(a**2 * np.exp(eps) * (4 + a * (a * np.exp(eps) - 4)))
    ) / (8 + 2 * a * (a * np.exp(eps) + a - 4))
    p_rr = 1 - rho
    q = rho
    scale_rr = 2 * p_rr - 1

    # Laplace parameters
    eps_laplace_val = eps - 4 * np.log(gamma)
    scale_lap_val = 1 / eps_laplace_val

    betas_rr = [compute_beta_rr(alpha, q, scale_rr, n) for alpha in alphas]
    betas_laplace = [compute_beta_laplace(alpha, scale_lap_val) for alpha in alphas]

    results_rr.append(betas_rr)
    results_laplace.append(betas_laplace)

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(9, 5))
cmap_rr = plt.cm.Blues(np.linspace(0.4, 1, len(epsilons)))
cmap_lap = plt.cm.Greens(np.linspace(0.4, 1, len(epsilons)))

for i, eps in enumerate(epsilons):
    # Solid lines
    plt.plot(
        alphas, results_rr[i], color=cmap_rr[i], linewidth=3, label=f"SOTA ε={eps:.1f}"
    )
    plt.plot(
        alphas,
        results_laplace[i],
        linestyle="-",
        linewidth=3,
        color=cmap_lap[i],
        label=f"Ours ε={eps:.1f}",
    )

    # Markers at evaluated points
    plt.scatter(alphas, results_rr[i], color=cmap_rr[i], marker="x", s=70, label=None)
    plt.scatter(
        alphas,
        results_laplace[i],
        color=cmap_lap[i],
        marker="o",
        s=70,
        edgecolors="black",
        label=None,
    )

plt.xlabel(r"$\alpha$", fontsize=20)
plt.ylabel(r"$\beta$", fontsize=20)
plt.title(r"$(\alpha,\beta)$-accuracy", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.grid(True, linestyle="-", alpha=0.6)
plt.legend(fontsize=14, loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()

# Save and show
filename = f"alfa_beta{a}_n{n}.png"
plt.savefig(filename, dpi=300)
plt.show()
