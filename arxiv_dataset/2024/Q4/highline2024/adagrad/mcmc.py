import matplotlib.pyplot as plt
import numpy as np


def mcmc(samples, f, l_min, l_max, proposal_sd=1.0, burn_in=1000, thinning=10, initial_value=1.0):
    np.random.seed(42)
    x = initial_value
    accepted_samples = []
    i = 0
    while len(accepted_samples) < samples:
        x_candidate = np.random.normal(x, proposal_sd)

        if x_candidate <= l_min or x_candidate >= l_max:
            continue

        # Compute the acceptance ratio
        acceptance_ratio = f(x_candidate) / f(x)

        # Accept or reject the candidate
        if np.random.uniform(0, 1) < acceptance_ratio:
            x = x_candidate

        # Skip the burn-in phase and apply thinning
        if i >= burn_in and (i - burn_in) % thinning == 0:
            accepted_samples.append(x)

        i += 1

    return np.array(accepted_samples)


if __name__ == "__main__":
    # Parameters
    def f(x): return np.exp(-x**2)
    l_min = -10
    l_max = 10
    samples = mcmc(100000, f, l_min, l_max)
    plt.hist(samples, bins=100, density=True)
    plt.show()
