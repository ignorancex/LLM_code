"""
----------------------------------------------------------------------
Project: water-demand-decomposition-sgd
Filename: composite_and_fit_model.py
Author: Roy Elkayam
Created: 2025-05-23
Description: this module contains the function to fit a composite model

----------------------------------------------------------------------
"""


import numpy as np
from scipy.optimize import minimize
from skewed_gaussian import skewed_gaussian


# --- Function to create the composite model ---
def composite_model(x, peaks, C_base, amplitudes, sigmas, alphas):
    """Composite model of multiple skewed Gaussians with fixed peaks."""
    result = np.full(len(x), C_base)
    for i, peak in enumerate(peaks):
        component = skewed_gaussian(x, amplitudes[i], peak, sigmas[i], alphas[i])
        result += component
    return result


# --- Function to fit model ensuring it passes through peaks ---
def fit_with_peak_constraints(x, y, peaks, pattern=None):
    """Fit model ensuring it passes through all detected peaks while optimizing all parameters."""
    n_peaks = len(peaks)
    C_base = np.min(y)

    # Initial guess for parameters
    amplitudes_init = [max(0.1, y[peak] - C_base) for peak in peaks]
    sigmas_init = [2.0] * n_peaks
    alphas_init = [0.0] * n_peaks

    # Function to optimize (sum of squared errors)
    def objective(params):
        C_base = params[0]
        amplitudes = params[1:n_peaks + 1]
        sigmas = params[n_peaks + 1:2 * n_peaks + 1]
        alphas = params[2 * n_peaks + 1:]

        model_y = composite_model(x, peaks, C_base, amplitudes, sigmas, alphas)
        # Regular MSE for points not at peaks
        mse = np.mean((y - model_y) ** 2)

        # Add regularization to prevent extreme values
        sigma_reg = 0.01 * np.sum((sigmas - 2.0) ** 2)  # Prefer sigmas near 2.0
        alpha_reg = 0.01 * np.sum(alphas ** 2)  # Prefer alphas near 0

        return mse + sigma_reg + alpha_reg

    # Constraint: model must pass through peaks
    def peak_constraints(params):
        C_base = params[0]
        amplitudes = params[1:n_peaks + 1]
        sigmas = params[n_peaks + 1:2 * n_peaks + 1]
        alphas = params[2 * n_peaks + 1:]

        model_y = composite_model(x, peaks, C_base, amplitudes, sigmas, alphas)
        constraints = []
        for i, peak in enumerate(peaks):
            # Constraint: model value at peak should equal actual value
            constraints.append(model_y[peak] - y[peak])

        return constraints

    # Set up optimization
    initial_params = [C_base] + amplitudes_init + sigmas_init + alphas_init

    # Define bounds for parameters
    bounds = [(0, np.max(y))]  # C_base bounds
    bounds += [(0, np.max(y) * 2)] * n_peaks  # Amplitude bounds
    bounds += [(0.1, 10)] * n_peaks  # Sigma bounds
    bounds += [(-5, 5)] * n_peaks  # Alpha bounds

    # Define constraints for scipy.optimize.minimize
    constraints = [
        {'type': 'eq', 'fun': lambda params: peak_constraints(params)[i]}
        for i in range(n_peaks)
    ]

    # Perform optimization with multiple attempts if needed
    best_result = None
    best_objective = float('inf')

    # Try different starting points for sigma and alpha
    sigma_starts = [1.0, 2.0, 3.0]
    alpha_starts = [-1.0, 0.0, 1.0]

    for sigma_start in sigma_starts:
        for alpha_start in alpha_starts:
            # Update initial parameters
            sigmas_init = [sigma_start] * n_peaks
            alphas_init = [alpha_start] * n_peaks
            initial_params = [C_base] + amplitudes_init + sigmas_init + alphas_init

            try:
                result = minimize(
                    objective,
                    initial_params,
                    method='SLSQP',  # Sequential Least Squares Programming
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-6, 'maxiter': 1000}
                )

                # Check if this is better than previous attempts
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
            except Exception as e:
                print(f"Optimization attempt failed: {e}")
                continue

    # If all attempts failed, try without constraints
    if best_result is None:
        # print(f"Warning: Constrained optimization failed. Trying unconstrained...")
        print(f"Warning: Constrained optimization (force through all peaks) using SLSQP failed. \n Trying unconstrained....\n  using L-BFGS-B: for pattern{pattern} where initial params were: {initial_params}")

        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 1000}
        )
        best_result = result

    # Extract optimized parameters
    params = best_result.x
    C_base = params[0]
    amplitudes = params[1:n_peaks + 1]
    sigmas = params[n_peaks + 1:2 * n_peaks + 1]
    alphas = params[2 * n_peaks + 1:]

    # Verify that the model passes through peaks
    model_y = composite_model(x, peaks, C_base, amplitudes, sigmas, alphas)
    peak_errors = [abs(model_y[peak] - y[peak]) for peak in peaks]
    max_peak_error = max(peak_errors)

    print(f"Max error at peaks: {max_peak_error:.4f}")
    if max_peak_error > 0.01:
        print("Warning: Model does not pass exactly through all peaks!")

    return C_base, amplitudes, sigmas, alphas




def fit_pattern(x, y, significant_peaks, pattern=None):
    """
    Fit a model to the data with constraints on peaks.

    Parameters:
    - x: Array of indices.
    - y: Array of data values.
    - significant_peaks: Array of peak indices.

    Returns:
    - C_base, amplitudes, sigmas, alphas: Model parameters.
    """
    return fit_with_peak_constraints(x, y, significant_peaks, pattern)
