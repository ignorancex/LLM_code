import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


fs = 22
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Set up the x-axis
x = np.linspace(-10, 10, 1000)

# Prior
prior_mu, prior_sigma = 0, 2
prior = gaussian(x, prior_mu, prior_sigma)

# Likelihood
likelihood_mu, likelihood_sigma = 5, 1.5
likelihood = gaussian(x, likelihood_mu, likelihood_sigma)

# Posterior (unnormalized)
posterior_unnorm = prior * likelihood

# Calculate evidence (normalization factor)
evidence = np.trapz(posterior_unnorm, x)

# Normalize the posterior
posterior = posterior_unnorm / evidence

# Calculate means and variances
posterior_mu = np.sum(x * posterior) / np.sum(posterior)
posterior_var = np.sum((x - posterior_mu)**2 * posterior) / np.sum(posterior)
posterior_sigma = np.sqrt(posterior_var)

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(x, prior, color='blue')
ax.plot(x, likelihood, color='red')
ax.plot(x, posterior, color='green')

#ax.set_title("Visualization of Bayes' Rule with Gaussian Distributions", fontsize=fs)
#ax.set_xlabel('x', fontsize=fs)
#ax.set_ylabel('Probability Density', fontsize=fs)
#ax.xtick_params([])#axis='both', which='major', labelsize=fs)
ax.grid(True, alpha=0.3)

# Add annotations with arrows
ax.annotate(f'Prior (μ={prior_mu}, σ={prior_sigma})', 
            xy=(prior_mu, gaussian(prior_mu, prior_mu, prior_sigma)), 
            xytext=(prior_mu-3, 0.28),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            color='blue', fontsize=fs)

ax.annotate(f'Likelihood (μ={likelihood_mu}, σ={likelihood_sigma})', 
            xy=(likelihood_mu, gaussian(likelihood_mu, likelihood_mu, likelihood_sigma)), 
            xytext=(likelihood_mu-1, 0.32),
            arrowprops=dict(facecolor='red', shrink=0.05),
            color='red', fontsize=fs)

# Get the position for the posterior annotation
xlim = ax.get_xlim()
ylim = ax.get_ylim()
posterior_x = (xlim[1] + prior_mu) / 2
posterior_y = (ylim[1] + 0.28) / 2  # 0.28 is the y-coordinate of the prior annotation

ax.annotate(f'Posterior (μ={posterior_mu:.2f}, σ={posterior_sigma:.2f})', 
            xy=(posterior_mu, max(posterior)), 
            xytext=(posterior_x-12, posterior_y),
            arrowprops=dict(facecolor='green', shrink=0.05),
            color='green', fontsize=fs)

# Print normalization factors and evidence
print(f"Prior normalization factor: {np.trapz(prior, x):.4f}")
print(f"Likelihood normalization factor: {np.trapz(likelihood, x):.4f}")
print(f"Posterior normalization factor: {np.trapz(posterior, x):.4f}")
print(f"Evidence: {evidence:.4f}")

plt.tight_layout()
plt.show()
