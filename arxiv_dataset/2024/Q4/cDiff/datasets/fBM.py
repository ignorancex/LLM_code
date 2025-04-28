import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import wishart, multivariate_normal
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader
from scipy.special import logit, expit, gammainccinv  # expit = sigmoid = inverse logit

scale_factor = 10.

import numpy as np
import cupy as cp

# JIT compile the function using Numba
import numpy as np

def compute_covariance(time_steps, hurst, tau2):
    """
    Compute the covariance matrix using CuPy for GPU acceleration.

    Parameters:
    - time_steps: Array of time steps (CuPy array)
    - hurst: Hurst parameter
    - tau2: Scaling factor for the covariance matrix

    Returns:
    - cov_matrix: Covariance matrix computed on GPU
    """
    # Convert time_steps to a CuPy array if not already
    time_steps = cp.asarray(time_steps)

    # Create a vectorized time step difference matrix (|t_i - t_j|) using CuPy
    time_diffs = cp.abs(cp.subtract.outer(time_steps, time_steps))  # Shape: (n, n)

    # Compute covariance matrix using broadcasting and vectorized operations on GPU
    time_steps_power = cp.power(time_steps, 2 * hurst)  # Shape: (n,)
    cov_matrix = 0.5 * (time_steps_power[:, None] + time_steps_power[None, :] - cp.power(time_diffs, 2 * hurst))

    # Scale by tau2
    cov_matrix *= tau2

    return cov_matrix

def simulate_fbm_with_drift(n, hurst, tau2=1, amplitude=1, period=1, phase=0, t=1, jitter=1e-6):
    """
    Simulate fractional Brownian motion with a parametrized sinusoidal drift term using GPU acceleration with CuPy.

    Parameters:
    - n: number of time steps
    - hurst: Hurst parameter (0 < H < 1)
    - amplitude: amplitude of the sinusoidal drift
    - period: period of the sinusoidal drift
    - phase: phase shift of the sinusoidal drift
    - t: total time (default is 1)
    - jitter: small value added to the diagonal to ensure the matrix is positive definite

    Returns:
    - time_steps: the time steps used in the simulation
    - fbm_with_drift: simulated fractional Brownian motion with sinusoidal drift
    """
    dt = t / n
    time_steps = np.linspace(0, t, n + 1)

    # Compute covariance matrix using Numba
    cov_matrix = compute_covariance(time_steps, hurst, tau2)

    # Convert the matrix to a CuPy array for GPU acceleration
    cov_matrix_gpu = cp.array(cov_matrix)

    # Add jitter to ensure the matrix is positive definite (GPU operation)
    cov_matrix_gpu += jitter * cp.eye(n + 1)

    # Perform Cholesky decomposition on GPU
    chol_gpu = cp.linalg.cholesky(cov_matrix_gpu)

    # Generate Gaussian random variables using CuPy (on GPU)
    normal_random_variables_gpu = cp.random.normal(0, 1, n + 1)

    # Compute the fBm path on the GPU
    fbm_path_gpu = cp.dot(chol_gpu, normal_random_variables_gpu)

    # Transfer the fBm path back to CPU memory
    fbm_path = cp.asnumpy(fbm_path_gpu)

    # Compute the sinusoidal drift term on the CPU
    drift = amplitude * np.sin((2 * np.pi * time_steps / period) + phase)

    # Add drift to the fBm path
    fbm_with_drift = fbm_path + drift

    return time_steps, fbm_with_drift
#
#
# def simulate_fbm_with_drift(n, hurst, tau2=1, amplitude=1, period=1, phase=0, t=1, jitter=1e-6):
#     """
#     Simulate fractional Brownian motion with a parametrized sinusoidal drift term.
#
#     Parameters:
#     - n: number of time steps
#     - hurst: Hurst parameter (0 < H < 1)
#     - amplitude: amplitude of the sinusoidal drift
#     - period: period of the sinusoidal drift
#     - phase: phase shift of the sinusoidal drift
#     - t: total time (default is 1)
#     - jitter: small value added to the diagonal to ensure the matrix is positive definite
#
#     Returns:
#     - time_steps: the time steps used in the simulation
#     - fbm_with_drift: simulated fractional Brownian motion with sinusoidal drift
#     """
#     dt = t / n
#     time_steps = np.linspace(0, t, n + 1)
#
#     # Create a vectorized time step difference matrix
#     time_diffs = np.abs(np.subtract.outer(time_steps, time_steps))  # (|t_i - t_j|)
#
#     # Compute the covariance matrix using broadcasting
#     cov_matrix = 0.5 * (np.power(time_steps[:, None], 2 * hurst) +
#                         np.power(time_steps[None, :], 2 * hurst) -
#                         np.power(time_diffs, 2 * hurst))
#
#     cov_matrix *= tau2
#
#     # Add jitter to ensure the matrix is positive definite
#     cov_matrix += jitter * np.eye(n + 1)
#
#     # Generate correlated Gaussian random variables using Cholesky decomposition
#     chol = np.linalg.cholesky(cov_matrix)
#     normal_random_variables = np.random.normal(0, 1, n + 1)
#     fbm_path = np.dot(chol, normal_random_variables)
#
#     # Compute the sinusoidal drift term
#     drift = amplitude * np.sin((2 * np.pi * time_steps / period) + phase)
#
#     # Add drift to the fBm path
#     fbm_with_drift = fbm_path + drift
#
#     return time_steps, fbm_with_drift


# Global hyperparams dictionary for prior hyperparameters
hyperparams = {
    'hurst_alpha': 1.0,  # Alpha for Beta distribution (controls the shape of Hurst exponent)
    'hurst_beta': 1.0,  # Beta for Beta distribution
    'tau2_alpha': 20.0,  # Alpha for Gamma distribution of variance tau2
    'tau2_beta': 1.0,  # Beta for Gamma distribution of variance tau2
    'amplitude_alpha': 3.0,  # Alpha for Gamma distribution of amplitude
    'amplitude_beta': 0.2,  # Beta for Gamma distribution of amplitude
    'phase_min': 0.0,  # Min phase (for Uniform distribution)
    'phase_max': 2 * np.pi  # Max phase (for Uniform distribution)
}



def sample_theta():
    """
    Sample the parameters (theta) for the fBm with sinusoidal drift model and return them without domain restrictions.
    """

    # Sample Hurst exponent (0 < hurst < 1) and apply logit transform
    hurst = logit(np.random.beta(hyperparams['hurst_alpha'], hyperparams['hurst_beta']))

    # Sample variance tau2 using Gamma distribution and apply log transform
    tau2 = np.log(np.random.gamma(hyperparams['tau2_alpha'], hyperparams['tau2_beta']))

    # Sample amplitude using Gamma distribution and apply log transform
    amplitude = np.log(np.random.gamma(hyperparams['amplitude_alpha'], hyperparams['amplitude_beta']))

    # Sample phase (0 <= phase < 2 * pi), map to real line
    phase = np.tan((np.random.uniform(hyperparams['phase_min'], hyperparams['phase_max']) - np.pi) / 2)

    # Sample period between 0.1 and 1 (Uniform distribution)
    period = np.random.uniform(0.1, 1)

    # Return the sampled parameters as a dictionary
    theta = {
        'hurst': hurst / scale_factor,
        'tau2': tau2 / scale_factor,
        'amplitude': amplitude / scale_factor,
        'phase': phase / scale_factor,
        'period': period
    }

    return theta


def sample_y(theta, n):
    """
    Wrapper around simulate_fbm_with_drift that accepts sampled parameters and undoes the transformations.

    Parameters:
    - theta: dictionary of sampled parameters from sample_theta
    - n: sample size (number of time steps)

    Returns:
    - time_steps: array of time steps
    - fbm_with_drift: array of simulated fractional Brownian motion with drift
    """

    # Undo the logit transformation for hurst to map back to [0, 1]
    hurst = expit(theta['hurst'] * scale_factor)

    # Undo the log transformation for tau2 and amplitude
    tau2 = np.exp(theta['tau2'] * scale_factor)
    amplitude = np.exp(theta['amplitude'] * scale_factor)

    # Undo the phase transformation to map back to [0, 2 * pi]
    phase = 2 * np.arctan(theta['phase'] * scale_factor) + np.pi

    # Extract period
    period = theta['period']

    # Call the simulate_fbm_with_drift function with the transformed parameters
    time_steps, fbm_with_drift = simulate_fbm_with_drift(
        n=n,
        hurst=hurst,
        tau2=tau2,
        amplitude=amplitude,
        period=period,
        phase=phase
    )

    return fbm_with_drift.reshape(-1,1)


def return_dl_ds(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False, num_workers=4):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl


if __name__ == "__main__":
    # Parameters
    n_batches = 1
    batch_size = 128

    import time
    import numpy as np
    import cupy as cp
    seed = 1024
    np.random.seed(seed)  # Set seed for NumPy (CPU operations)
    cp.random.seed(seed)

    # previousely 23 second for 5 batch
    # now 9 seconds

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=1000, return_ds=True)

    start_time = time.time()
    # Iterate over the DataLoader
    for theta, y in dl:
        # Assuming theta contains mu and Sigma, and y contains the observed data
        print("Theta", theta[:10])  # Print first two parameter samples
        print("Data shape:", y.shape)
        print("---")
    esp_time = time.time() - start_time
    print(esp_time)
    import matplotlib.pyplot as plt

    time_steps = np.linspace(0, 1, 1000 + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, y[100].cpu().numpy())
    plt.title(f'Simulated fBm with Sinusoidal Drift (n={1000})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("fBM.png")