"""
----------------------------------------------------------------------
Project: water-demand-decomposition-sgd
Filename: symmetric_gaussian_baseline.py
Author: Roy Elkayam
Created: 2025-05-23
Description: skewed Gaussian function for symmetric baseline fitting
----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize

# Skewed Gaussian function
def skewed_gaussian(x, A, mu, sigma, alpha):
    norm_gaussian = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew

# Symmetric Gaussian fitting function (alpha = 0)
def fit_symmetric_gaussian_baseline(x, y, peaks):
    n_peaks = len(peaks)
    C_base = np.min(y)
    amplitudes_init = [max(0.1, y[peak] - C_base) for peak in peaks]
    sigmas_init = [2.0] * n_peaks

    def symmetric_model(x, C_base, amplitudes, sigmas):
        result = np.full(len(x), C_base)
        for i, peak in enumerate(peaks):
            component = skewed_gaussian(x, amplitudes[i], peak, sigmas[i], alpha=0)
            result += component
        return result

    def objective(params):
        C_base = params[0]
        amplitudes = params[1:n_peaks + 1]
        sigmas = params[n_peaks + 1:]
        model_y = symmetric_model(x, C_base, amplitudes, sigmas)
        return np.mean((y - model_y) ** 2) + 0.01 * np.sum((sigmas - 2.0) ** 2)

    initial_params = [C_base] + amplitudes_init + sigmas_init
    bounds = [(0, np.max(y))] + [(0, np.max(y) * 2)] * n_peaks + [(0.1, 10)] * n_peaks

    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 1000}
    )

    if not result.success:
        print("Baseline optimization failed.")
        return None

    params = result.x
    C_base = params[0]
    amplitudes = params[1:n_peaks + 1]
    sigmas = params[n_peaks + 1:]
    alphas = [0.0] * n_peaks  # Forced symmetry

    return C_base, amplitudes, sigmas, alphas

# Evaluation and plotting
def evaluate_and_plot(x, y, peaks, C_base, amplitudes, sigmas, alphas, label='Model'):
    predicted = np.full(len(x), C_base)
    for i, peak in enumerate(peaks):
        predicted += skewed_gaussian(x, amplitudes[i], peak, sigmas[i], alphas[i])

    # Volume conservation
    scale_factor = np.trapezoid(y, dx=1) / np.trapezoid(predicted, dx=1)
    predicted *= scale_factor

    # Metrics
    rmse = np.sqrt(np.mean((y - predicted) ** 2))
    mae = np.mean(np.abs(y - predicted))
    max_error = np.max(np.abs(y - predicted))
    r2 = 1 - np.sum((y - predicted)**2) / np.sum((y - np.mean(y))**2)

    plt.plot(x, y, label='Actual', marker='o')
    plt.plot(x, predicted, '--', label=label)
    plt.title(f"{label} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
    plt.legend()
    plt.grid(True)

    return {'RMSE': rmse, 'MAE': mae, 'Max Error': max_error, 'R²': r2}

# Example usage
if __name__ == "__main__":
    # Example synthetic dataset
    hours = np.arange(24)
    y = np.array([
        10, 12, 18, 25, 22, 20, 18, 15,
        13, 12, 11, 10, 10, 11, 13, 18,
        20, 22, 25, 23, 18, 15, 13, 11
    ])
    peaks = [3, 17]  # Roughly morning and evening

    # Fit symmetric baseline
    baseline = fit_symmetric_gaussian_baseline(hours, y, peaks)
    if baseline:
        C_b, A_b, sigma_b, alpha_b = baseline
        plt.figure(figsize=(10, 4))
        metrics_baseline = evaluate_and_plot(hours, y, peaks, C_b, A_b, sigma_b, alpha_b, label="Symmetric Baseline")

    plt.show()
