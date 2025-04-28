import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import wishart, multivariate_normal, invwishart
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader
import cupy as cp

y_dim = 4  # Dimensionality of the multivariate normal

# Hyperparameters for the NIW prior
niw_hyperpar_dict = {
    "mu_0": np.zeros(y_dim),             # Prior mean for mu
    "kappa_0": 1.0,                     # Scaling factor for the mean
    "nu_0": y_dim + 2,                  # Degrees of freedom for the inverse-Wishart (must be > y_dim - 1)
    "Psi_0": np.eye(y_dim)               # Scale matrix for the inverse-Wishart distribution
}


# Helper functions to convert between Sigma and its Cholesky components (diagonal and off diagonal)

# def disassemble_sigma(Sigma):
#     """
#     Disassemble the covariance matrix into its Cholesky decomposition components:
#     log of the diagonal and the off-diagonal elements.
#
#     Parameters:
#     Sigma (ndarray): Covariance matrix.
#
#     Returns:
#     dict: Dictionary containing log of the diagonal elements and the off-diagonal elements
#           of the Cholesky factor.
#     """
#     # Compute the Cholesky decomposition
#     L = np.linalg.cholesky(Sigma)
#
#     # Extract diagonal and off-diagonal elements
#     L_diag = np.diag(L)  # Diagonal elements
#     L_offdiag = L[np.tril_indices(L.shape[0], k=-1)]  # Off-diagonal elements (below diagonal)
#
#     # Return log of the diagonal and the off-diagonal elements
#     return {
#         "log_L_diag": np.log(L_diag),
#         "L_offdiag": L_offdiag
#     }

def disassemble_sigma(Sigma):
    """
    Disassemble the covariance matrix into its Cholesky decomposition components
    using CuPy: log of the diagonal and the off-diagonal elements.

    Parameters:
    Sigma (cp.ndarray): Covariance matrix.

    Returns:
    dict: Dictionary containing log of the diagonal elements and the off-diagonal elements
          of the Cholesky factor.
    """
    # Compute the Cholesky decomposition using CuPy
    L = cp.linalg.cholesky(Sigma)

    # Extract diagonal and off-diagonal elements using CuPy methods
    L_diag = cp.diag(L)  # Diagonal elements
    L_offdiag = L[cp.tril_indices(L.shape[0], k=-1)]  # Off-diagonal elements (below diagonal)

    # Return log of the diagonal and the off-diagonal elements
    return {
        "log_L_diag": cp.log(L_diag),
        "L_offdiag": L_offdiag
    }

#
# def reassemble_sigma(log_L_diag, L_offdiag):
#     """
#     Reassemble the covariance matrix from the log of the diagonal elements and the off-diagonal elements
#     of the Cholesky factor.
#
#     Parameters:
#     log_L_diag (array-like): Log of the diagonal elements of the Cholesky factor.
#     L_offdiag (array-like): Off-diagonal elements of the Cholesky factor (flattened).
#
#     Returns:
#     Sigma (ndarray): Reconstructed covariance matrix.
#     """
#     # Infer the dimension from the length of log_L_diag
#     y_dim = len(log_L_diag)
#
#     # Initialize an empty lower triangular matrix for the Cholesky factor
#     L = np.zeros((y_dim, y_dim))
#
#     # Fill the diagonal elements by exponentiating the log of the diagonal elements
#     L[np.diag_indices(y_dim)] = np.exp(log_L_diag)
#
#     # Fill the off-diagonal elements (below the diagonal)
#     L[np.tril_indices(y_dim, k=-1)] = L_offdiag
#
#     # Reassemble the covariance matrix using the Cholesky factor
#     Sigma = L @ L.T  # Equivalent to L dot L^T (L times its transpose)
#
#     return Sigma

def reassemble_sigma(log_L_diag, L_offdiag):
    """
    Reassemble the covariance matrix from the log of the diagonal elements and the off-diagonal elements
    of the Cholesky factor using CuPy.

    Parameters:
    log_L_diag (cp.ndarray or array-like): Log of the diagonal elements of the Cholesky factor.
    L_offdiag (cp.ndarray or array-like): Off-diagonal elements of the Cholesky factor (flattened).

    Returns:
    cp.ndarray: Reconstructed covariance matrix.
    """
    # Infer the dimension from the length of log_L_diag
    y_dim = len(log_L_diag)

    # Initialize an empty lower triangular matrix for the Cholesky factor using CuPy
    L = cp.zeros((y_dim, y_dim))

    # Fill the diagonal elements by exponentiating the log of the diagonal elements
    L[cp.diag_indices(y_dim)] = cp.exp(log_L_diag)

    # Fill the off-diagonal elements (below the diagonal)
    L[cp.tril_indices(y_dim, k=-1)] = L_offdiag

    # Reassemble the covariance matrix using the Cholesky factor
    Sigma = L @ L.T  # Equivalent to L dot L^T (L times its transpose)

    return Sigma


# Unit test

def test_disassemble_reassemble():
    # Construct a covariance matrix (Sigma)
    y_dim = 4

    A = np.random.randn(y_dim, y_dim)
    Sigma = A @ A.T  # Construct a positive definite covariance matrix

    # Disassemble Sigma into log_L_diag and L_offdiag
    components = disassemble_sigma(Sigma)

    # Reassemble Sigma from the components
    reassembled_Sigma = reassemble_sigma(components["log_L_diag"], components["L_offdiag"])

    # Check if the original Sigma and reassembled Sigma are approximately equal
    if np.allclose(Sigma, reassembled_Sigma, atol=1e-8):
        print("Test passed: Reassembled Sigma matches the original Sigma.")
    else:
        print("Test failed: Reassembled Sigma does not match the original Sigma.")

    # Print the matrices for visual comparison (optional)
    print("Original Sigma:\n", Sigma)
    print("Reassembled Sigma:\n", reassembled_Sigma)

#
# def sample_NIW_parameters(input_dict):
#     """
#     Sample mu and Sigma from the Normal-Inverse-Wishart (NIW) prior and disassemble Sigma.
#
#     Parameters:
#     input_dict (dict): Dictionary containing the NIW hyperparameters.
#
#     Returns:
#     dict: Dictionary containing the sampled mean vector (mu), log_L_diag, and L_offdiag.
#     """
#
#     # Extract hyperparameters
#     mu_0 = input_dict["mu_0"]
#     kappa_0 = input_dict["kappa_0"]
#     nu_0 = input_dict["nu_0"]
#     Psi_0 = input_dict["Psi_0"]
#
#     # Sample precision matrix (inverse covariance) from the Wishart distribution
#     precision_matrix = wishart.rvs(df=nu_0, scale=Psi_0)
#
#     # Invert the precision matrix to get the covariance matrix (Sigma)
#     Sigma = np.linalg.inv(precision_matrix)
#
#     # Sample the mean mu from the normal distribution given Sigma
#     mu = np.random.multivariate_normal(mu_0, Sigma / kappa_0)
#
#     # Disassemble Sigma into its Cholesky components
#     cholesky_components = disassemble_sigma(Sigma)
#
#     return {
#         "mu": mu,
#         "log_L_diag": cholesky_components["log_L_diag"],
#         "L_offdiag": cholesky_components["L_offdiag"]
#     }

def sample_NIW_parameters(input_dict):
    """
    Sample mu and Sigma from the Normal-Inverse-Wishart (NIW) prior using CuPy and disassemble Sigma.

    Parameters:
    input_dict (dict): Dictionary containing the NIW hyperparameters.

    Returns:
    dict: Dictionary containing the sampled mean vector (mu), log_L_diag, and L_offdiag.
    """

    # Extract hyperparameters from the input dictionary
    mu_0 = input_dict["mu_0"]            # Mean vector of the NIW distribution
    kappa_0 = input_dict["kappa_0"]       # Scaling factor
    nu_0 = input_dict["nu_0"]             # Degrees of freedom for the Wishart distribution
    Psi_0 = input_dict["Psi_0"]           # Scale matrix for the Wishart distribution

    # Convert Psi_0 to a CuPy array if it isn't already
    Psi_0 = cp.asarray(Psi_0)

    # Sample precision matrix (inverse covariance) from the Wishart distribution using SciPy
    # precision_matrix = wishart.rvs(df=nu_0, scale=cp.asnumpy(Psi_0))
    # Sigma = cp.linalg.inv(cp.asarray(precision_matrix))

    Sigma = invwishart.rvs(df=nu_0, scale=cp.asnumpy(Psi_0))

    # Sample the mean vector mu from the normal distribution given Sigma
    # Note: Convert the covariance matrix to CPU (NumPy) for random sampling since CuPy doesn't have a direct multivariate_normal sampler
    mu = np.random.multivariate_normal(cp.asnumpy(mu_0), Sigma / kappa_0)

    # Disassemble Sigma into its Cholesky components using the previous function
    cholesky_components = disassemble_sigma(cp.asarray(Sigma))

    return {
        "mu": mu,  # Already a NumPy array from sampling
        "log_L_diag": cholesky_components["log_L_diag"].get(),
        "L_offdiag": cholesky_components["L_offdiag"].get()
    }



# Function to sample parameters from the NIW prior
def sample_theta():
    return sample_NIW_parameters(niw_hyperpar_dict)


# Function to sample data from the multivariate normal using the sampled NIW parameters
# def sample_y(parms_dict, sample_size):
#     """
#     Generate data from the multivariate normal distribution using the sampled mu and reassembled Sigma.
#
#     Parameters:
#     parms_dict (dict): Dictionary containing the mean (mu), log_L_diag, and L_offdiag.
#     sample_size (int): Number of data points to generate.
#
#     Returns:
#     np.ndarray: Generated data of shape (sample_size, y_dim).
#     """
#
#     mu = parms_dict["mu"]
#     log_L_diag = parms_dict["log_L_diag"]
#     L_offdiag = parms_dict["L_offdiag"]
#
#     # Reassemble the covariance matrix Sigma from the Cholesky components
#     Sigma = reassemble_sigma(log_L_diag, L_offdiag)
#
#     # Generate multivariate normal data
#     samples = np.random.multivariate_normal(mu, Sigma, size=sample_size)
#
#     return samples


def sample_y(parms_dict, sample_size):
    """
    Generate data from the multivariate normal distribution using the sampled mu and reassembled Sigma
    using CuPy for GPU-accelerated operations.

    Parameters:
    parms_dict (dict): Dictionary containing the mean (mu), log_L_diag, and L_offdiag.
    sample_size (int): Number of data points to generate.

    Returns:
    cp.ndarray: Generated data of shape (sample_size, y_dim) in CuPy format.
    """

    # Extract parameters from the dictionary
    mu = cp.asarray(parms_dict["mu"])  # Convert mu to CuPy array
    log_L_diag = cp.asarray(parms_dict["log_L_diag"])  # Already a CuPy array if using previous implementation
    L_offdiag = cp.asarray(parms_dict["L_offdiag"])    # Already a CuPy array if using previous implementation

    # Reassemble the covariance matrix Sigma from the Cholesky components
    Sigma = reassemble_sigma(log_L_diag, L_offdiag)

    # Generate data from the multivariate normal distribution
    # CuPy currently does not have a multivariate normal sampler, so use NumPy for sampling
    mu_cpu = cp.asnumpy(mu)              # Convert mu to NumPy array for sampling
    Sigma_cpu = cp.asnumpy(Sigma)        # Convert Sigma to NumPy array for sampling

    # Use NumPy to generate samples since it supports multivariate normal sampling
    samples_cpu = np.random.multivariate_normal(mu_cpu, Sigma_cpu, size=sample_size)

    return samples_cpu


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

# Example usage
if __name__ == "__main__":
    # Parameters
    n_batches = 2
    batch_size = 128

    import time

    start_time = time.time()
    # Generate DataLoader and Dataset
    dl, ds = return_dl_ds(n_batches=n_batches, batch_size=batch_size, n_sample=1000, return_ds=True)

    # Iterate over the DataLoader
    for theta, y in dl:
        # Assuming theta contains mu and Sigma, and y contains the observed data
        print("Theta", theta[:10])  # Print first two parameter samples
        # print("Data shape:", y[:10])
        print("---")
    print(time.time() - start_time)
