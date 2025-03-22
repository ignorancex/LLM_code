import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
np.random.seed(42)
X = np.linspace(-5, 5, 100)
true_slope = 0.7
true_intercept = 1.5
y = true_slope * X + true_intercept + np.random.normal(0, 1, size=X.shape)

# Define the Bayesian linear regression model
tfd = tfp.distributions

# Define priors
prior_slope = tfd.Normal(loc=0., scale=1.)
prior_intercept = tfd.Normal(loc=0., scale=1.)
prior_sigma = tfd.HalfNormal(scale=1.)

# Define likelihood function
def likelihood(slope, intercept, sigma, X):
    mean = slope * X + intercept
    return tfd.Normal(loc=mean, scale=sigma)

# Sample from the posterior using Markov Chain Monte Carlo (MCMC)
@tf.function
def joint_log_prob(slope, intercept, sigma):
    lp = prior_slope.log_prob(slope) + prior_intercept.log_prob(intercept) + prior_sigma.log_prob(sigma)
    lp += tf.reduce_sum(likelihood(slope, intercept, sigma, X).log_prob(y))
    return lp

# Initialize MCMC sampler
initial_state = [0., 0., 1.]
num_results = 1000
kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=joint_log_prob,
    step_size=0.1,
    num_leapfrog_steps=3)

# Run MCMC
states, kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    current_state=initial_state,
    kernel=kernel,
    trace_fn=lambda _, pkr: pkr.is_accepted)

# Extract sampled parameters
slope_samples, intercept_samples, sigma_samples = states

# Plot the posterior distributions
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].hist(slope_samples, bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Posterior of Slope')
axs[1].hist(intercept_samples, bins=30, color='skyblue', edgecolor='black')
axs[1].set_title('Posterior of Intercept')
axs[2].hist(sigma_samples, bins=30, color='skyblue', edgecolor='black')
axs[2].set_title('Posterior of Sigma')

plt.show()