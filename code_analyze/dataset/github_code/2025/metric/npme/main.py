import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from elliptical_sampler import EllipticalSliceSampler

def sigmoid(x):
    """Sigmoid function to map values between 0 and 1."""
    return 1 / (1 + np.exp(-x))

# Step 1: Define a New Two-Dimensional Non-linear Function
def nonlinear_function(x1, x2):
    """
    Computes a non-linear function of x1 and x2.
    
    Args:
    - x1 (np.array): First input array.
    - x2 (np.array): Second input array.
    
    Returns:
    - np.array: Computed function values.
    """
    return np.log(1 + x1) + np.log(1 + x2)

# Generate a 2D grid of points
x1 = np.linspace(0, 1, 20)
x2 = np.linspace(0, 1, 20)
x1_grid, x2_grid = np.meshgrid(x1, x2)
x_grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
f_values = nonlinear_function(x_grid_points[:, 0], x_grid_points[:, 1])

# Step 2: Generate Preferences Using Bradley-Terry Model Over the Grid
def generate_preferences(f_vals, num_prefs=10000):
    """
    Generates preferences based on the Bradley-Terry model.
    
    Args:
    - f_vals (np.array): Function values at grid points.
    - num_prefs (int): Number of preference pairs to generate.
    
    Returns:
    - list of tuple: Generated preference pairs (i, j).
    """
    preferences = []
    num_points = len(f_vals)
    for _ in range(num_prefs):
        i, j = np.random.choice(num_points, size=2, replace=False)
        # Probability of preference using Bradley-Terry model
        p_ij = sigmoid(f_vals[i] - f_vals[j])
        # Decide preference based on random draw
        if np.random.rand() < p_ij:
            preferences.append((i, j))
        else:
            preferences.append((j, i))
    return preferences

preferences = generate_preferences(f_values)

# Step 3: Define the Likelihood Function for Elliptical Slice Sampling
def loglik_from_preferences(f):
    """
    Log-likelihood function using Bradley-Terry model for preferences.
    
    Args:
    - f (np.array): Sampled function values.
    
    Returns:
    - float: Log-likelihood value.
    """
    log_lik = 0
    for idx_i, idx_j in preferences:
        # YOUR CODE HERE (~2 lines)
        log_lik += np.log(sigmoid(f[idx_i] - f[idx_j]))
        # END OF YOUR CODE
    return log_lik

# Step 4: Define the RBF Kernel to Compute Prior Covariance Matrix
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Computes the Radial Basis Function (RBF) kernel between two sets of points.
    
    Args:
    - X1, X2 (np.array): Input data points.
    - length_scale (float): Kernel length scale parameter.
    - sigma_f (float): Kernel output scale.
    
    Returns:
    - np.array: RBF kernel matrix.
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Define prior covariance (prior mean is zero vector)
sigma_prior = rbf_kernel(x_grid_points, x_grid_points, length_scale=1.0, sigma_f=1.0)

# Add small jitter to diagonal for numerical stability
jitter = 1e-6
sigma_prior += jitter * np.eye(sigma_prior.shape[0])

# Ensure the matrix is symmetric to avoid numerical issues
sigma_prior = (sigma_prior + sigma_prior.T) / 2

# Step 5: Run Elliptical Slice Sampling
sampler = EllipticalSliceSampler(sigma_prior, loglik_from_preferences)
samples = sampler.sample(1000, n_burn=500)
average_samples = np.mean(samples, axis=0)

# Generate true function values on grid points
true_values_on_grid = nonlinear_function(x_grid_points[:, 0], x_grid_points[:, 1])

def evaluate_elicited_metric(true_metric, elicited_metric):
    """
    Evaluates and prints the mean and standard deviation of the difference
    between true and elicited metrics.
    
    Args:
    - true_metric (np.array): True values of the function.
    - elicited_metric (np.array): Elicited (estimated) function values.
    """
    # YOUR CODE HERE
    diffs = true_metric - elicited_metric
    print(np.mean(diffs), np.std(diffs))
    # END OF YOUR CODE

evaluate_elicited_metric(true_values_on_grid, average_samples)

# Step 6: Plot the True Non-linear Function and Elicited Metric in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the true function
x1_fine = np.linspace(0, 1, 50)
x2_fine = np.linspace(0, 1, 50)
x1_fine_grid, x2_fine_grid = np.meshgrid(x1_fine, x2_fine)
true_f_values = nonlinear_function(x1_fine_grid, x2_fine_grid)
ax.plot_surface(x1_fine_grid, x2_fine_grid, true_f_values, color='blue', alpha=0.5, label='True Function')

# Plot the averaged samples as a surface
x1_avg = x_grid_points[:, 0].reshape(20, 20)
x2_avg = x_grid_points[:, 1].reshape(20, 20)
avg_values = average_samples.reshape(20, 20)
ax.plot_surface(x1_avg, x2_avg, avg_values, color='red', alpha=0.5, label='Estimated Function')

# Customize plot
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('True Function vs. Averaged Estimated Function')
plt.legend()
plt.show()
