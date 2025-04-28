import numpy as np
from numba import njit
from torch.utils.data import DataLoader, Dataset
from .BayesDataStream import BayesDataStream

scale_y = 100.
scale_sigma = 3.
def get_companion_matrix(Gamma):
    'Gamma must be an ndarray of shape [L,L,max_lag]'
    L, _, max_lag = Gamma.shape

    # Create the companion matrix
    companion_matrix = np.zeros((L * max_lag, L * max_lag))
    companion_matrix[:L, :] = np.column_stack([Gamma[:, :, h] for h in range(max_lag)])
    companion_matrix[L:, :-L] = np.eye(L * (max_lag - 1))
    return companion_matrix

def is_var_stationary(Gamma):
    A = get_companion_matrix(Gamma)
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(A)
    # Check if all eigenvalues have absolute values less than 1
    return np.all(np.abs(eigenvalues) < 1)

def shrink_VAR_to_stationary(Gamma, shrink=0.99, max_iter=1000):
    assert 0 < shrink < 1, "shrink must be between 0 and 1, noninclusive."
    L, _, max_lag = Gamma.shape
    current_Gamma = Gamma.copy()
    iter_count = 0

    while not is_var_stationary(current_Gamma) and iter_count < max_iter:
        current_Gamma = shrink * current_Gamma
        iter_count += 1

    if iter_count == max_iter:
        print("Warning: Maximum iterations reached. Stationarity not guaranteed.")

    return current_Gamma


@njit
def sample_minnesota_prior(L, max_lag, tau=0.5, theta=1.0, stationary=True):
    if max_lag < 1:
        raise ValueError("max_lag should be a positive integer.")

    Gamma = np.zeros((L, L, max_lag))

    for i in range(L):
        for j in range(L):
            for h in range(1, max_lag + 1):
                mean = (i == j) * tau / h
                std = theta / h
                draw = np.random.normal(mean, std)
                while np.abs(draw) > 1:
                    draw = np.random.normal(mean, std)
                Gamma[i, j, h - 1] = draw

    # Omit stationary adjustment for simplicity
    return Gamma


@njit
def sample_VAR_parameters(L, max_lag, tau, theta, sigma_shape, sigma_scale, stationary):
    # Sample Gamma from the Minnesota prior
    Gamma = sample_minnesota_prior(L, max_lag, tau=tau, theta=theta, stationary=stationary)

    # Sample diagonal elements of Sigma from the inverse gamma prior
    Sigma = 1.0 / np.random.gamma(sigma_shape, 1 / sigma_scale, L)

    return Gamma, Sigma


@njit
def sample_VAR_data(Gamma, Sigma, seq_length, initial_values):
    L, _, max_lag = Gamma.shape

    # Check if the initial_values has the correct shape
    if initial_values.shape != (max_lag, L):
        raise ValueError("initial_values must have shape (max_lag, L)")

    data = np.zeros((L, seq_length))

    # Fill in the first max_lag columns with the transpose of initial_values
    data[:, :max_lag] = initial_values.T

    # Simulate the VAR model
    for t in range(max_lag, seq_length):
        lagged_values = np.zeros(L)
        for h in range(max_lag):
            lagged_values += Gamma[:, :, h] @ data[:, t - h - 1]

        # Generate noise assuming Sigma is a vector of variances
        noise = np.zeros(L)
        for i in range(L):
            noise[i] = np.random.normal(0.0, np.sqrt(Sigma[i]))  # Fix for Numba compatibility

        data[:, t] = lagged_values + noise

    return data.transpose()


# Wrapper to sample the parameters theta (Gamma and Sigma)
def sample_theta():
    L = 5
    max_lag = 3
    tau = 0.45
    theta = 0.25
    sigma_shape = 2.0
    sigma_scale = 1.0 / 6.0
    stationary = True

    # Sample VAR parameters using the previously defined Numba-compiled function
    Gamma, Sigma = sample_VAR_parameters(L, max_lag, tau, theta, sigma_shape, sigma_scale, stationary)
    Gamma = shrink_VAR_to_stationary(Gamma)

    # Return a dictionary containing the sampled parameters
    return {"Gamma": Gamma, "Sigma": np.log(Sigma) / scale_sigma}


# Wrapper to sample VAR data using theta_dict and N (number of samples to generate)
def sample_y(theta_dict, N):
    Gamma = theta_dict["Gamma"]
    Sigma = np.exp(theta_dict["Sigma"] * scale_sigma)

    # Extract VAR dimension (L) and lag order (max_lag) from Gamma
    L, _, max_lag = Gamma.shape

    # Set initial values for the simulation (could also be passed as an argument)
    initial_values = np.zeros((max_lag, L))  # Default initial values (zeros)

    # Sample VAR data using the previously defined Numba-compiled function
    data = sample_VAR_data(Gamma, Sigma, N, initial_values)

    return data / scale_y



# Function to generate DataLoader and Dataset
def return_dl_ds(n_batches=256, batch_size=128, n_sample=None, return_ds=False, num_workers=4):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=250, high=1000):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl, ds
    else:
        return dl

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # theta_dict = sample_theta()
    # T = 1000
    # # Simulate the VAR process
    # y = sample_y(theta_dict, T)
    # print(y.shape)
    # plt.plot(y)
    # plt.savefig("m.png")

    n_batches = 20
    batch_size = 128

    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=None, return_ds=True)
    n_theta_samples = 10000
    # Iterate over the DataLoader
    for theta, y in dl:
        # print(y.shape)
        print(y.min())
        #print(theta.min())