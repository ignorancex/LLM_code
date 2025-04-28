import numpy as np
import matplotlib.pyplot as plt
from numpy.random import gamma, dirichlet, exponential, normal
from torch.utils.data import DataLoader, Dataset
from .BayesDataStream import BayesDataStream

scale_tau = 4.
scale_mu = 50.
scale_y = 100
def inverse_gamma(shape, rate):
    """ Sample from an Inverse Gamma with shape, rate parameterization. """
    return 1 / gamma(shape, 1 / rate)


def sample_theta():
    """
    Samples theta_i from the Dirichlet-Laplace prior with updated Dirichlet, tau priors,
    and an inverse gamma prior on sigma^2.

    Returns:
        dict: A dictionary containing sampled 'theta', 'tau', 'lambda', 'delta', and 'sigma'.
    """
    # Hyperparameters
    a = global_hyperparameters['a']
    b = global_hyperparameters['b']
    N = global_hyperparameters['N']
    sigma2_shape = global_hyperparameters['sigma2_shape']
    sigma2_rate = global_hyperparameters['sigma2_rate']

    # Global shrinkage parameter tau ~ Gamma(n * a, b / 2)
    tau = gamma(N * a, 2 / b)

    # Local shrinkage parameters lambda_i ~ Exp(1) for i in 1,...,N
    lambda_vals = exponential(1, N)

    # Dirichlet-distributed weights delta ~ Dir(a, a, ..., a)
    delta_vals = dirichlet([a] * N)

    # Compute mu_i = tau * delta_i * lambda_i
    local_scales = tau * delta_vals * lambda_vals
    mu = normal(0.0, local_scales)

    # Sample sigma^2 from an Inverse Gamma distribution with shape=8, rate=8
    sigma2 = inverse_gamma(sigma2_shape, sigma2_rate)
    sigma = np.sqrt(sigma2)  # Convert variance to standard deviation

    return {
        'mu': np.tanh(mu / scale_mu),
        'tau': np.log(tau) / scale_tau, # try log or other transformation
        'sigma': np.log(sigma)
    }


def sample_y(theta, n):
    """
    Simulates n replicated observations for each theta_i from the normal likelihood.

    Args:
        theta (dict): A dictionary containing 'theta' values (mean parameters).
        n (int): Number of replicated observations per mean (theta_i).

    Returns:
        np.ndarray: Simulated observations from the model, with shape (N, n),
                    where N is the number of theta values.
    """
    # Extract theta values from the dictionary
    mu = np.arctanh(theta['mu']) * scale_mu
    N = len(mu)

    # Sample y_ij ~ N(theta_i, sigma^2) where sigma is the global standard deviation
    sigma = np.exp(theta['sigma'])

    # Sample n replicates for each theta_i using broadcasting
    y_vals = normal(loc=mu[np.newaxis, :], scale=sigma, size=(n, N))#.squeeze()

    return y_vals / scale_y

n_signals = 20
# Global hyperparameters (example structure with updated b default)
global_hyperparameters = {
    'N': 100,
    'a': 1.0/np.sqrt(n_signals),        # Dirichlet concentration parameter
    'b': 1.0,        # Rate parameter for tau (default 1)
    'sigma2_shape': 100.0,
    'sigma2_rate': 100.0
}


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
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
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

if __name__ == "__main__":
    # Sample theta
    # theta = sample_theta()
    #
    # # Sample observations
    # y = sample_y(theta, 1)
    #
    # # Display the results
    #
    # # Generate a stem plot to represent the size of each theta
    # plt.figure(figsize=(8, 5))
    # plt.stem(range(global_hyperparameters['N']), theta['mu'], linefmt='b-', markerfmt='bo', basefmt='r-')
    # plt.xlabel('Index of Theta')
    # plt.ylabel('Theta Value')
    # plt.title('Stem Plot of Sampled Theta Values')
    # plt.grid(True)
    # plt.savefig("dirichlet_laplace.png")

    n_batches = 10
    batch_size = 128

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=1, return_ds=True)
    n_theta_samples = 10000
    # Iterate over the DataLoader
    for theta, y in dl:
        # print(y.shape)
        print(y.max())
        print(y.min())