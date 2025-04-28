import torch

import math
import numpy.linalg as linalg
from dataclasses import dataclass
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

from multiprocessing.dummy import Pool
from itertools import repeat

# Define hyperparameters
hyperparms = {
    'beta1_mean': 1.0,
    'beta1_tau': 1.0,
    'beta2_mean': -1.0,
    'beta2_tau': 1.0,
    'sigma_y_shape': 2.0,
    'sigma_y_scale': 2.0,
    'sigma_x_shape': 4.0,
    'sigma_x_scale': 1.0,
    'p1_a': 30.0,
    'p1_b': 2.0,
    'p2_a': 30.0,
    'p2_b': 2.0
}

scale_factor = 1.

# simple markov chain class
class MarkovChainFinite:
    def __init__(self, Q: np.ndarray):
        qshape = Q.shape
        if len(qshape) != 2 or qshape[0] != qshape[1]:
            raise ValueError('Transition matrix Q must be a square 2D array.')
        qrowsums = np.sum(Q, axis=1)
        if not np.all(qrowsums == 1.):
            raise ValueError('Transition matrix Q must be a row stochastic matrix.')
        self.n_states = int(qshape[0])
        self.Q = Q

    def sample_trajectory(self, num_steps: int, initial_state=None):
        if initial_state is not None:
            self.initial_state = initial_state
        else:
            # roughly approximate the stationary distribution
            v = np.ones(self.n_states) / float(self.n_states)
            Qn = linalg.matrix_power(self.Q, 25)
            pi_stationary = v @ Qn
            # draw the initial state from the approximate stationary distribution
            self.initial_state = np.random.choice(self.n_states, p=pi_stationary)

        x = np.zeros(num_steps, dtype=np.int8)
        x[0] = self.initial_state

        cumulative_Q = np.cumsum(self.Q, axis=1)

        # Generate random numbers to determine transitions
        random_values = np.random.rand(num_steps - 1)

        # Use binary search to find the corresponding state for each transition
        for t in range(1, num_steps):
            x[t] = np.searchsorted(cumulative_Q[x[t - 1]], random_values[t - 1])

        # for t in range(1, num_steps):
        #     x[t] = np.random.choice(self.n_states, p=self.Q[x[t - 1], :])
        # x[1:] = 1

        return x


@dataclass
class AR1:
    phi: float
    sigma: float

    def sample_trajectory(self, num_steps, initial_state=None):
        x = np.zeros(num_steps)
        if initial_state is not None:
            x[0] = initial_state
        elif abs(self.phi) >= 1.0:
            x[0] = 0.0
        else:
            # simulate from the marginal distribution
            x[0] = np.random.normal(loc=0.0, scale=self.sigma / np.sqrt(1.0 - self.phi * self.phi))

        # here is the random walk
        # for t in range(1, num_steps):
        #     x[t] = x[t - 1] + np.random.normal(loc=0.0, scale=self.sigma)

        random_steps = np.random.normal(loc=0.0, scale=self.sigma, size=num_steps - 1)
        x[1:] = np.cumsum(random_steps) + x[0]
        return x


@dataclass
class MarkovSwitchingFactorModel:
    beta1: float
    beta2: float
    log_sigma_y: float
    log_sigma_x: float
    logit_p1: float
    logit_p2: float

    def __post_init__(self):
        self.beta = np.array([self.beta1, self.beta2])
        self.sigma_y = math.exp(self.log_sigma_y)
        self.sigma_x = math.exp(self.log_sigma_x)
        self.p1 = 1.0 / (1.0 + math.exp(-self.logit_p1 * scale_factor))
        self.p2 = 1.0 / (1.0 + math.exp(-self.logit_p2 * scale_factor))
        self.Q = np.array([[self.p1, 1.0 - self.p1], [1.0 - self.p2, self.p2]])
        self.lambda_chain = MarkovChainFinite(self.Q)
        self.x_chain = AR1(phi=1.0, sigma=self.sigma_x)

    def sample_trajectory(self, num_steps, lam0=None, x0=None):
        lam_traj = self.lambda_chain.sample_trajectory(num_steps, lam0)
        beta_traj = self.beta[lam_traj]
        beta_traj = np.row_stack(np.broadcast_arrays(1.0, beta_traj))
        x_traj = self.x_chain.sample_trajectory(num_steps, x0)
        y_traj = np.random.normal(loc=x_traj * beta_traj, scale=self.sigma_y).transpose()
        return y_traj, x_traj, lam_traj

# Wrap a sample_theta_MSFM function to use our dictionary of hyperparameters
def sample_theta():
    """
    Sample the parameters for the Markov Switching Factor Model (MSFM) based on the given hyperparameters.

    Parameters:
    hyperparms (dict): Dictionary containing the hyperparameters for sampling MSFM parameters.

    Returns:
    dict: Dictionary containing the sampled parameters (theta).
    """

    # Extract hyperparameters from the dictionary
    beta1_mean = hyperparms.get('beta1_mean', 1.0)
    beta1_tau = hyperparms.get('beta1_tau', 1.0)
    beta2_mean = hyperparms.get('beta2_mean', -1.0)
    beta2_tau = hyperparms.get('beta2_tau', 1.0)
    sigma_y_shape = hyperparms.get('sigma_y_shape', 2.0)
    sigma_y_scale = hyperparms.get('sigma_y_scale', 2.0)
    sigma_x_shape = hyperparms.get('sigma_x_shape', 4.0)
    sigma_x_scale = hyperparms.get('sigma_x_scale', 1.0)
    p1_a = hyperparms.get('p1_a', 30.0)
    p1_b = hyperparms.get('p1_b', 2.0)
    p2_a = hyperparms.get('p2_a', 30.0)
    p2_b = hyperparms.get('p2_b', 2.0)

    # Observation and latent factor variances
    sigma_y = 1.0 / np.sqrt(np.random.default_rng().gamma(shape=sigma_y_shape, scale=sigma_y_scale))
    sigma_x = 1.0 / np.sqrt(np.random.default_rng().gamma(shape=sigma_x_shape, scale=sigma_x_scale))

    # Factor loadings under the two states
    beta1 = np.random.normal(beta1_mean, beta1_tau * sigma_y)
    beta2 = np.random.normal(beta2_mean, beta2_tau * sigma_y)

    # Enforce that beta1 <= beta2 for identifiability
    if beta1 > beta2:
        beta1, beta2 = beta2, beta1

    # Transition probabilities
    p1 = 0.99 * np.random.beta(p1_a, p1_b)
    p2 = 0.99 * np.random.beta(p2_a, p2_b)

    # Construct the output dictionary of parameters
    theta = {
        'beta1': beta1,
        'beta2': beta2,
        'log_sigma_y': math.log(sigma_y),
        'log_sigma_x': math.log(sigma_x),
        'logit_p1': math.log(p1 / (1.0 - p1)) / scale_factor,
        'logit_p2': math.log(p2 / (1.0 - p2)) / scale_factor
    }

    return theta


def sample_y(theta, seq_length):
    """
    Sample the observed data (y) from the Markov Switching Factor Model (MSFM) based on sampled parameters.

    Parameters:
    theta (dict): Dictionary containing the model parameters.
    seq_length (int): The length of the trajectory to sample.

    Returns:
    np.ndarray: Sampled y data of shape (seq_length, y_dim).
    """

    # Initialize the Markov Switching Factor Model with the given parameters
    y_model = MarkovSwitchingFactorModel(**theta)

    # Sample the trajectory
    y_traj, _, _ = y_model.sample_trajectory(seq_length)

    return y_traj

def return_dl_ds(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False, num_workers=4):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=200, high=500):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl


if __name__ == "__main__":
    # Parameters
    n_batches = 1
    batch_size = 128

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=1000, return_ds=True)

    import time

    start_time = time.time()
    # Iterate over the DataLoader
    for theta, y in dl:
        # Assuming theta contains the sampled theta from Gamma distribution
        print("Theta shape", theta[:20])  # Print first two parameter samples
        print("Data shape:", y.shape)
        print("---")
    print(time.time() - start_time)

    y_data = y[0].squeeze()
    # Create a time axis
    time_steps = np.arange(len(y_data))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, y_data)

    # Add labels and title
    plt.xlabel('Time step')
    plt.ylabel('Observed value (y)')
    plt.title('Sampled Markov Switching Factor Model Data Over Time')
    plt.grid(True)

    # Show the plot
    plt.savefig("markov_switch.png")
