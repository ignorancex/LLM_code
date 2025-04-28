import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import multivariate_normal, norm
from scipy.special import logit, expit  # expit = sigmoid = inverse logit
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader


# Hyperparameters
n_species = 3
lambda_total = 50  # Average total number of birds observed in each survey
alpha_dirichlet = 2.0 * np.ones(n_species)  # Prior for Dirichlet distribution


# function to simulate observed counts with Poisson assumption and imperfect detection
def simulate_species_counts(true_p, detection_prob, lambda_total, num_surveys):
    # Simulate true counts using Poisson for each species
    # the total number of birds in the area is N ~ Poisson(lambda_total)
    # then conditional on N, we have multinomial sampling.
    true_counts = np.random.poisson(lambda_total * true_p, size=(num_surveys, len(true_p)))

    # Simulate observed counts with imperfect detection
    observed_counts = np.random.binomial(true_counts, detection_prob)

    return observed_counts / lambda_total * 2 - 0.5


# Define mixture model parameters for the first species' detection probability
p_mixture = 0.7  # Probability of "easy detection"
detection_prob_easy = 0.9  # Easy detection probability
detection_prob_hard = 0.3  # Hard detection probability

# Detection probabilities for the other two species (fixed)
detection_prob_species_2 = 0.6
detection_prob_species_3 = 0.7


# Function to sample detection probabilities with a mixture model for species 1
def sample_detection_prob():
    # Draw detection probability for the first species from a mixture model
    if np.random.rand() < p_mixture:
        detection_prob_1 = detection_prob_easy  # Easy detection
    else:
        detection_prob_1 = detection_prob_hard  # Hard detection

    # Return detection probabilities for all species
    detection_prob = np.array([detection_prob_1, detection_prob_species_2, detection_prob_species_3])
    return detection_prob


# wrapper functions

def my_gen_sample_size(n, low=10, high=50):
    return np.random.randint(low=low, high=high, size=n)


# Function to sample theta (true proportions)
def sample_theta():
    # Draw from Dirichlet prior for species proportions
    true_p = np.random.dirichlet(alpha_dirichlet)
    baseline = true_p[-1]
    true_p = true_p - baseline
    true_p = true_p[:-1]
    return {'true_p': true_p}


# Sample y function, using the mixture model for species 1 detection probability
def sample_y(parms, num_surveys):
    # Sample detection probabilities with mixture model for the first species
    detection_prob = sample_detection_prob()
    true_p = parms['true_p']

    baseline = (1 - np.sum(true_p)) / n_species
    true_p = true_p + baseline
    true_p = np.append(true_p, baseline)
    if np.sum(true_p) != 1.:
        true_p = true_p / np.sum(true_p)
    return simulate_species_counts(true_p, detection_prob, lambda_total, num_surveys)


def return_species_dl(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False, num_workers=4):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=200, high=500):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl


if __name__ == "__main__":
    dl, ds = return_species_dl(n_batches = 2,batch_size = 128, return_ds=True)

    for theta, y in dl:
        print(y[:10])