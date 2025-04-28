import numpy as np
from torch.utils.data import DataLoader, Dataset
from .BayesDataStream import BayesDataStream
from scipy.stats import dirichlet
import torch
import matplotlib.pyplot as plt
from scipy.stats import beta

K = 5
n_multi = 300
alpha = np.random.gamma(shape=5, scale=0.5, size=K)
#alpha = np.array([1.43658871,2.84772868,5.67357521,2.2688366 ,3.45225569])
print(alpha)

def sample_theta():
    """
    Generates alpha from a Gamma prior and samples theta from a Dirichlet distribution.

    Returns:
        dict: A dictionary containing 'alpha' and 'theta'.
    """
    # Sample theta from Dirichlet(alpha)
    theta = np.random.dirichlet(alpha + 1e-5)
    baseline = theta[-1]
    theta = theta - baseline

    theta = theta[:-1]
    return {'theta': theta}

# Function to sample y (observed counts)
def sample_y(parms, n):
    """
    Generates multinomial counts given theta and sample size n.

    Parameters:
        parms (dict): Parameters containing 'theta'.
        n (int): Total number of data points.

    Returns:
        dict: A dictionary containing 'counts'.
    """
    theta = parms['theta']
    baseline = (1 - np.sum(theta)) / K
    theta = theta + baseline
    theta = np.append(theta, baseline)
    if np.sum(theta) != 1.:
        theta = theta / np.sum(theta)
    counts = np.random.multinomial(n_multi, theta, size=n)
    counts = counts / n_multi
    return counts

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


def close_form_distance(y, theta_samples):
    """
    Plots histograms and density functions for Dirichlet-distributed samples.

    Parameters:
    y (numpy.ndarray): Observation data of shape [n_samples, n_dim].
    theta_samples (numpy.ndarray or torch.Tensor): Sampled theta values of shape [n_samples, n_dim].

    Returns:
    float: Log density of the theta samples under the Dirichlet distribution.
    """
    # Assertions to check if y and theta_samples have 2 dimensions
    assert y.ndim == 2, "y should have 2 dimensions"
    assert theta_samples.ndim == 2, "theta_samples should have 2 dimensions"

    # Convert tensors to numpy arrays if necessary
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(theta_samples, torch.Tensor):
        theta_samples = theta_samples.detach().cpu().numpy()

    # Compute the baseline for theta_samples and normalize
    baseline = (1 - np.sum(theta_samples, axis=1)) / K
    theta_samples = theta_samples + baseline.reshape(-1, 1)
    theta_samples = np.hstack((theta_samples, baseline.reshape(-1, 1)))
    theta_samples = theta_samples / np.sum(theta_samples, axis=1, keepdims=True)

    # Compute alpha_n using the observations
    alpha_n = alpha + (y * n_multi).sum(axis=0).astype(int)  # Update Dirichlet parameter

    # Define Dirichlet distribution and compute log density
    D = torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha_n, dtype=torch.float32))
    log_density = D.log_prob(torch.tensor(theta_samples, dtype=torch.float32)).sum().item()

    return log_density

def plot_dirichlet_margin(y, theta_samples, save_path, zoom=True):
    # Plot histogram and density for each dimension of theta_samples
    assert y.ndim == 2, "y should have 2 dimensions"
    assert theta_samples.ndim == 2, "theta_samples should have 2 dimensions"

    # Convert tensors to numpy arrays if necessary
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(theta_samples, torch.Tensor):
        theta_samples = theta_samples.detach().cpu().numpy()

    # Compute the baseline for theta_samples and normalize
    baseline = (1 - np.sum(theta_samples, axis=1)) / K
    theta_samples = theta_samples + baseline.reshape(-1, 1)
    theta_samples = np.hstack((theta_samples, baseline.reshape(-1, 1)))
    theta_samples = theta_samples / np.sum(theta_samples, axis=1, keepdims=True)

    # Compute alpha_n using the observations
    alpha_n = alpha + (y * n_multi).sum(axis=0).astype(int)  # Update Dirichlet parameter
    n_theta_samples = theta_samples.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(50, 10))
    for i in range(K):
        # Plot histogram of the i-th dimension of theta_samples
        axes[i].hist(theta_samples[:, i], bins=30, density=True, alpha=0.6, color='blue', label='Theta Samples')

        # Calculate and plot density function of beta(alpha_n[i], sum(alpha_n) - alpha_n[i])
        a_i = alpha_n[i]
        b_i = alpha_n.sum() - alpha_n[i]
        if zoom:
            mean = a_i / (a_i + b_i)
            var = a_i * b_i / ((a_i + b_i) ** 2 * (a_i + b_i + 1))
            interval_length = 4 * np.sqrt(var)
            x = np.linspace(mean - interval_length, mean + interval_length, 100)
        else:
            x = np.linspace(0, 1, 100)
        beta_density = beta(a_i, b_i).pdf(x)
        axes[i].plot(x, beta_density, 'r-', lw=2, label=f'Beta({a_i:.2f}, {b_i:.2f})')

        # Add titles and legends
        axes[i].set_title(f'Dimension {i + 1}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/dirichlet_margin.png")

# Example usage
if __name__ == "__main__":
    # Parameters
    n_batches = 2
    batch_size = 10

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=1000, return_ds=True)
    n_theta_samples = 10000
    # Iterate over the DataLoader
    for theta, y in dl:
        # print("Counts:", y[0, :10,])
        print("Theta", theta[:2])
        print("Theta shape:", theta.shape)
        print("data shape:", y.shape)
        alpha_n = alpha + np.int32(y[0] * n_multi).sum(axis=0)
        D = torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha_n))
        theta = D.rsample((n_theta_samples,))
        # print(D.log_prob(torch.tensor(theta)).sum().item())
        # print(close_form_distance(y[0].squeeze(), theta))
        plot_dirichlet_margin(y[0].squeeze(), theta, ".")
        print("---")

