import numpy as np
from torch.utils.data import DataLoader, Dataset
from .BayesDataStream import BayesDataStream
import torch
from scipy.stats import gamma, poisson

# Parameters for the Gamma distribution
alpha = 2.0  # Shape parameter
beta = 1.0  # Rate parameter (inverse scale)
theta_dim = 10

# Function to sample from Gamma distribution to get theta
def sample_theta():
    """
    Samples theta from a Gamma distribution.

    Returns:
        dict: A dictionary containing 'theta' sampled from the Gamma distribution.
    """
    theta = gamma.rvs(a=alpha, scale=1.0 / beta, size=theta_dim)
    theta = np.log(theta)
    # return {'theta': theta,"theta1":theta}
    return {'theta': theta}


# Function to sample y (observed counts) from a Poisson distribution given theta
def sample_y(parms, n):
    """
    Generates Poisson counts given theta (which can be a vector) and the number of samples.

    Parameters:
        parms (dict): Parameters containing 'theta' (vector).
        n (int): Number of samples to generate per element in theta.

    Returns:
        np.ndarray: Array of sampled Poisson counts.
    """
    theta = parms['theta']

    # Ensure theta is a vector and exponentiate it
    theta = np.exp(theta)

    # For each theta value, generate 'n' Poisson samples
    counts = np.array([poisson.rvs(mu=theta_i, size=n) for theta_i in theta])

    # Apply log transformation with a small constant to avoid log(0)
    counts = np.log(counts + 1)

    # Reshape the counts array
    return np.transpose(counts)


# Function to generate DataLoader and Dataset
def return_dl_ds(n_batches=256, batch_size=128, n_sample=None, return_ds=False, num_workers=4):
    """
    Creates a DataLoader and Dataset for batch processing.

    Parameters:
        n_batches (int): Number of batches.
        batch_size (int): Size of each batch.
        n_sample (int, optional): Fixed sample size for each data point.
        return_ds (bool): Whether to return the Dataset along with the DataLoader.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: PyTorch DataLoader for iterating over the dataset.
        Dataset (optional): The dataset used by the DataLoader.
    """
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample + 1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=200, high=500):
            return np.random.randint(low=low, high=high, size=n)

    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    ds.reset_batch_sample_sizes()

    if return_ds:
        return dl, ds
    else:
        return dl


# Function to compute posterior log density for Poisson-Gamma
def posterior_log_density(y, theta):
    """
    Computes the log density of Poisson given theta.

    Parameters:
        y (np.ndarray): Observed Poisson counts.
        theta (float): Mean parameter from Gamma distribution (Poisson rate).

    Returns:
        float: Log density of the observed data given the Poisson distribution with rate theta.
    """
    # Check if y and theta are tensors, and if so, convert to numpy arrays
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    y = np.exp(y).astype(int)
    theta = np.exp(theta)
    alpha_posterior = y.sum(0) + alpha
    beta_posterior = y.shape[0] + beta
    # Compute the log density for Poisson
    log_density = gamma.logpdf(theta, a=alpha_posterior, scale=1/beta_posterior).sum()

    return log_density


# Example usage
if __name__ == "__main__":
    # Parameters
    n_batches = 2
    batch_size = 128

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=None, return_ds=True)

    # Iterate over the DataLoader
    for theta, y in dl:
        # Assuming theta contains the sampled theta from Gamma distribution
        print("Theta shape", theta.shape)  # Print first two parameter samples
        print("Data shape:", y.shape)
        print(posterior_log_density(y[0], theta[0]))  # Log density for first sample
        print("---")
