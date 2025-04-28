import numpy as np
import sys
sys.path.append("../../")
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

# Dictionary of hyperparameters for the Witch's Hat distribution
hyperpar_dict = {
    "d": 5,  # Dimension
    "sigma": 0.02,  # Standard deviation of the Gaussian
    "delta": 0.05  # Weight of the uniform component
}

def sample_witch_hat(theta, n_samples, d, sigma, delta):
    # Determine the number of samples to draw from each component using a binomial distribution
    uniform_samples_count = np.random.binomial(n_samples, delta)
    gaussian_samples_count = n_samples - uniform_samples_count

    # Sample from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, (uniform_samples_count, d))

    # Sample from the multivariate Gaussian distribution
    gaussian_samples = np.random.multivariate_normal(theta, sigma ** 2 * np.eye(d), gaussian_samples_count)

    # Combine the samples
    samples = np.vstack((uniform_samples, gaussian_samples))

    # Shuffle the samples to ensure random mixing
    np.random.shuffle(samples)

    return samples

def sample_witch_hat_parameters(input_dict):
    """
    Sample parameters for the Witch's Hat distribution.

    Parameters:
    input_dict (dict): Dictionary containing the hyperparameters for the distributions.

    Returns:
    dict: Dictionary containing the sampled parameters.
    """

    # Extract hyperparameters
    d = input_dict["d"]
    sigma = input_dict["sigma"]
    delta = input_dict["delta"]

    # Sample the location parameter theta uniformly on the unit hypercube [0.1, 0.9]^d
    theta = np.random.uniform(0.1, 0.9, d)

    # Return the sampled parameters in a dictionary
    return {"theta": theta}



def my_gen_sample_size(n, low=100, high=1000):
    return np.random.randint(low=low, high=high, size=n)


def sample_theta():
    return sample_witch_hat_parameters(hyperpar_dict)


def sample_witch_hat_data(parms_dict, sample_size):
    """
    Generate samples from the Witch's Hat distribution using the sampled parameters.

    Parameters:
    parms_dict (dict): Dictionary containing the sampled parameters.
    sample_size (int): The number of samples to generate.

    Returns:
    np.array: Array of generated samples.
    """

    # Extract parameters
    theta = parms_dict["theta"]

    # Globally set hyperparameters
    d = hyperpar_dict["d"]
    sigma = hyperpar_dict["sigma"]
    delta = hyperpar_dict["delta"]

    # Determine the number of samples to draw from each component using a binomial distribution
    uniform_samples_count = np.random.binomial(sample_size, delta)
    gaussian_samples_count = sample_size - uniform_samples_count

    # Sample from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, (uniform_samples_count, d))

    # Sample from the multivariate Gaussian distribution
    gaussian_samples = np.random.multivariate_normal(theta, sigma ** 2 * np.eye(d), gaussian_samples_count)

    # Combine the samples
    samples = np.vstack((uniform_samples, gaussian_samples))

    # Shuffle the samples to ensure random mixing
    np.random.shuffle(samples)

    return samples

def return_witch_hat_dl(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_witch_hat_data, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl

if __name__ == "__main__":
    dl = return_witch_hat_dl(n_batches=2,batch_size=2,n_sample=1)
    for batch in dl:
        theta, y = batch
        print(theta.shape)
        print(y.shape)
