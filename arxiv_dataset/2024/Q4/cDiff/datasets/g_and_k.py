import sys
sys.path.append("../../")

from .BayesDataStream import BayesDataStream
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader
import torch

scale_odds_order = 5.
scale_even_order = 2.
scale_y = 10.

def generate_g_and_k_samples(a, b, g, k, num_samples, c=0.8):
    """
    Generate samples from the g-and-k distribution.

    Parameters:
    a (float): location parameter
    b (float): scale parameter
    g (float): skewness parameter
    k (float): kurtosis parameter
    c (float): additional parameter for the tanh function
    num_samples (int): number of samples to generate

    Returns:
    np.ndarray: samples from the g-and-k distribution
    """

    def g_and_k_quantile_function(a, b, g, k, c, u):
        z = norm.ppf(u)  # Get the quantile function of the standard normal distribution
        return a + b * (1 + c * np.tanh(g * z / 2)) * (1 + z ** 2) ** k * z

    # Generate uniform random variables
    u = np.random.uniform(0, 1, num_samples)

    # Apply the g-and-k quantile function
    samples = g_and_k_quantile_function(a, b, g, k, c, u)

    return samples


hyperpar_dict = {
    "a_mean": 0,
    "a_std": 1,
    "b_shape": 5,
    "b_scale": 1 / 5,
    "g_mean": 0,
    "g_std": 1,
    "k_shape": 7,
    "k_scale": 1 / 7
}

def sample_g_and_k_parameters(input_dict):
    """
    Sample parameters a, b, g, and k from their corresponding distributions.

    Parameters:
    input_dict (dict): Dictionary containing the hyperparameters for the distributions.

    Returns:
    dict: Dictionary containing the sampled parameters.
    """

    # Extract hyperparameters
    a_mean = input_dict["a_mean"]
    a_std = input_dict["a_std"]
    b_shape = input_dict["b_shape"]
    b_scale = input_dict["b_scale"]
    g_mean = input_dict["g_mean"]
    g_std = input_dict["g_std"]
    k_shape = input_dict["k_shape"]
    k_scale = input_dict["k_scale"]

    # Sample from the prior distributions
    # TODO: k should be positive. Can be Gamma(7,7)
    a = np.random.normal(loc=a_mean, scale=a_std)
    b = np.random.gamma(shape=b_shape, scale=b_scale)
    g = np.random.normal(loc=g_mean, scale=g_std)
    k = np.random.gamma(shape=k_shape, scale=k_scale) #

    # Return the sampled parameters in a dictionary
    return {"a": a / scale_odds_order, "log_b": np.log(b + 1e-3) / scale_even_order, "g": g / scale_odds_order, "log_k": np.log(k + 1e-3) / scale_even_order}


# wrapper functions to sample n, theta, y
def my_gen_sample_size(n, low=100, high=1000):
    return np.random.randint(low=low, high=high, size=n)


def sample_theta():
    return sample_g_and_k_parameters(hyperpar_dict)


def sample_g_and_k_data(parms_dict, sample_size):
    # Generate samples using the sampled parameters
    samples = generate_g_and_k_samples(
        a=parms_dict["a"] * scale_odds_order,
        b=np.exp(parms_dict["log_b"] * scale_even_order),
        g=parms_dict["g"] * scale_odds_order,
        k=np.exp(parms_dict["log_k"] * scale_even_order),
        num_samples=sample_size
    )
    samples = samples.reshape(-1, 1)
    return np.tanh(samples / scale_y)

def return_g_and_k_dl(n_batches=256,batch_size=128, n_sample=None, return_ds=False):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_g_and_k_data, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl

if __name__ == "__main__":
    dl = return_g_and_k_dl(n_batches=20)
    for batch in dl:
        theta, y = batch
        #print(theta[:20])
        #print(theta.min(axis=0))
        #print(theta.max(axis=0))
        print(y[:10])
        # print(y.max())
        # print(y.min())
