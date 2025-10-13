"""
----------------------------------------------------------------------
Project: water-demand-decomposition-sgd
Filename: composite_and_fit_model.py
Author: Roy Elkayam
Created: 2025-05-23
Description: this script implements a composite model for water demand

----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf

from cons import DefaultsParameters
from skewed_gaussian import skewed_gaussian
from composite_and_fit_model import composite_model, fit_pattern
from tools import find_significant_peaks


# === Configuration ===
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})


def save_model_parameters(results, filename):
    records = []
    for pattern, params in results.items():
        record = {
            'pattern': pattern,
            'N_peaks': len(params['peaks']),
            'C_base': round(params['C_base'], 2),
            'Amplitude_Min': round(min(params['amplitudes']), 2),
            'Amplitude_Max': round(max(params['amplitudes']), 2),
            'Sigma_Min': round(min(params['sigmas']), 2),
            'Sigma_Max': round(max(params['sigmas']), 2),
            'Alpha_Min': round(min(params['alphas']), 2),
            'Alpha_Max': round(max(params['alphas']), 2),
        }
        records.append(record)
    df = pd.DataFrame(records)
    df.to_excel(filename, index=False)
    print(f"Saved model parameters to {filename}")


def save_validation_metrics(df, results, filename):
    records = []
    for pattern in results:
        y_actual = df[pattern].values
        r = results[pattern]
        y_pred = composite_model(np.arange(24), r['peaks'], r['C_base'], r['amplitudes'], r['sigmas'], r['alphas'])
        y_pred *= np.trapezoid(y_actual, dx=1) / np.trapezoid(y_pred, dx=1)

        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mae = np.mean(np.abs(y_actual - y_pred))
        max_err = np.max(np.abs(y_actual - y_pred))
        r2 = 1 - np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)

        records.append({
            'Pattern': pattern,
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'Max Error': round(max_err, 2),
            'R²': round(r2, 3)
        })

    pd.DataFrame(records).to_excel(filename, index=False)
    print(f"Saved evaluation metrics to {filename}")




# === Utility Functions ===

def store_results(df, y, peaks, C_base, amplitudes, sigmas, alphas):
    try:
        peak_times = [df.index[p].strftime('%H:%M') for p in peaks]
    except Exception:
        peak_times = [None] * len(peaks)

    return {
        'peaks': peaks.tolist(),
        'peak_times': peak_times,
        'peak_values': y[peaks].tolist(),
        'C_base': C_base,
        'amplitudes': amplitudes.tolist(),
        'sigmas': sigmas.tolist(),
        'alphas': alphas
    }


def print_pattern_summary(pattern, peaks, C_base, sigmas, alphas):
    print(f"\n{pattern}: {len(peaks)} peaks")
    print(f"  Base consumption: {C_base:.2f}")
    print("  Widths (σ):", ", ".join([f"{s:.2f}" for s in sigmas]))
    print("  Skews (α):", ", ".join([f"{a:.2f}" for a in alphas]))


# === Model Fitting Functions ===

def fit_symmetric_model(x, y, peaks, pattern=None):
    n_peaks = len(peaks)
    C_base = np.min(y)
    A_init = [max(0.1, y[peak] - C_base) for peak in peaks]
    sig_init = [2.0] * n_peaks
    alp_init = [0.0] * n_peaks  # forced symmetric

    def objective(params):
        C_base = params[0]
        A = params[1:n_peaks + 1]
        σ = params[n_peaks + 1:2 * n_peaks + 1]
        model_y = composite_model(x, peaks, C_base, A, σ, alp_init)
        mse = np.mean((y - model_y) ** 2)
        reg = 0.01 * np.sum((σ - 2.0) ** 2)
        return mse + reg

    initial = [C_base] + A_init + sig_init + alp_init
    bounds = [(0, np.max(y))] + [(0, np.max(y) * 2)] * n_peaks + [(0.1, 10)] * n_peaks + [(0, 0)] * n_peaks

    try:
        result = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)
        if result.success:
            params = result.x
            C_base = params[0]
            A = params[1:n_peaks + 1]
            σ = params[n_peaks + 1:2 * n_peaks + 1]
            α = [0.0] * n_peaks
            return C_base, A, σ, α
        else:
            print(f"[{pattern}] Optimization failed.")
    except Exception as e:
        print(f"[{pattern}] Exception during optimization: {e}")

    return None, None, None, None


# === Decomposition ===

def decompose_patterns(df, patterns, symmetric=False):
    results = {}
    for pattern in patterns:
        print(f"\nProcessing: {pattern}")
        y = df[pattern].values
        x = np.arange(len(y))
        peaks = find_significant_peaks(df[[pattern]])

        if symmetric:
            C_base, A, σ, α = fit_symmetric_model(x, y, peaks, pattern)
        else:
            C_base, A, σ, α = fit_pattern(x, y, peaks, pattern)

        if C_base is not None:
            results[pattern] = store_results(df, y, peaks, C_base, A, σ, α)
            print_pattern_summary(pattern, peaks, C_base, σ, α)

    return results


# === Plotting ===

def plot_model_components(df, results, hours, time_labels, title_suffix=""):
    num_patterns = len(results)
    cols = 2
    rows = (num_patterns + cols - 1) // cols
    plt.figure(figsize=(14, 5 * rows))

    for i, pattern in enumerate(results, 1):
        actual = df[pattern].values
        r = results[pattern]
        peaks = np.array(r['peaks'])

        pred = composite_model(hours, peaks, r['C_base'], r['amplitudes'], r['sigmas'], r['alphas'])
        pred *= np.trapezoid(actual, dx=1) / np.trapezoid(pred, dx=1)

        plt.subplot(rows, cols, i)
        plt.plot(hours, actual, label='Actual', color='blue', marker='o')

        for j, peak in enumerate(peaks):
            component = skewed_gaussian(hours, r['amplitudes'][j], peak, r['sigmas'][j], r['alphas'][j])
            plt.plot(hours, r['C_base'] + component, linestyle='--', alpha=0.5,
                     label=f'Peak at {results[pattern]["peak_times"][j][:-3]}: ')

        plt.plot(hours, pred, label='Prediction', color='red')
        plt.ylabel(f"{pattern} {title_suffix}")
        # plt.title(f"{pattern} {title_suffix}")
        plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


# === Main ===

def main():
    df = pd.read_excel('demand_pattern.xlsx')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df.set_index('Time', inplace=True)

    patterns = df.columns.tolist()
    hours = np.arange(24)
    time_labels = [t.strftime('%H') for t in df.index]

    print("=== SGD Fit ===")
    sgd_results = decompose_patterns(df, patterns, symmetric=False)
    plot_model_components(df, sgd_results, hours, time_labels, title_suffix="[SGD]")

    print("=== Symmetric Gaussian Fit ===")
    sym_results = decompose_patterns(df, patterns, symmetric=True)
    plot_model_components(df, sym_results, hours, time_labels, title_suffix="[Symmetric]")

    # Optional: save parameters or evaluation metrics
    save_model_parameters(sgd_results, 'sgd_results.xlsx')
    save_model_parameters(sym_results, 'symmetric_results.xlsx')

    save_validation_metrics(df, sgd_results, 'sgd_validation_metrics.xlsx')
    save_validation_metrics(df, sym_results, 'symmetric_validation_metrics.xlsx')
    print("=== Saved! ===")

if __name__ == "__main__":
    main()
