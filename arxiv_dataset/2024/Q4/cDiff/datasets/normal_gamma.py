import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .BayesDataStream import BayesDataStream

mu=0.
kappa=1

d=8.
eta=8.

def sample_theta():
    sigma = np.sqrt(1./np.random.gamma(shape=d/2., scale = 2./eta))
    log_sigma = np.log(sigma)
    mu = np.random.normal(0, sigma/np.sqrt(kappa))
    return {'mu': mu, 'log_sigma':log_sigma}

def sample_y(parms, n):
    mu = parms['mu']
    sigma = np.exp(parms['log_sigma'])
    y = np.random.normal(mu, sigma, size=n)
    return y.reshape(-1, 1)

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

# Example usage
if __name__ == "__main__":
    # Parameters
    n_batches = 2
    batch_size = 128

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=None, return_ds=True)

    # Iterate over the DataLoader
    for theta, y in dl:
        # print("Counts:", y[0, :10,])
        print("Theta", theta[:2])
        print("Theta shape:", theta.shape)
        print("data shape:", y.shape)
        print("---")
