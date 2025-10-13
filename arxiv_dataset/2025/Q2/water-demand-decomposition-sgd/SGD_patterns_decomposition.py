"""
----------------------------------------------------------------------
Project: water-demand-decomposition-sgd
Filename: SGD_patterns_decomposition.py
Author: Roy Elkayam
Created: 2025-05-23
Description: this script is used to decompose water demand patterns and plot the results.

----------------------------------------------------------------------
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf

from cons import DefaultsParameters
from skewed_gaussian import skewed_gaussian
from composite_and_fit_model import composite_model, fit_pattern
from tools import find_significant_peaks



plt.rcParams.update({
    'font.size': 16,          # Default font size
    'axes.labelsize': 16,     # Font size for axis labels
    'axes.titlesize': 16,     # Font size for axis titles
    'xtick.labelsize': 16,    # Font size for x-axis tick labels
    'ytick.labelsize': 16,    # Font size for y-axis tick labels
    'legend.fontsize': 12,    # Font size for legend
    'figure.titlesize': 16    # Font size for figure titles
})


def store_results(df, y, significant_peaks, C_base, amplitudes, sigmas, alphas):
    """
    Store model results for a pattern.

    Parameters:
    - df: DataFrame with time-indexed data.
    - pattern: Name of the pattern.
    - y: Array of data values.
    - significant_peaks: Array of peak indices.
    - C_base, amplitudes, sigmas, alphas: Model parameters.

    Returns:
    - Dictionary with results.
    """
    peak_time = [None for _ in range(len(significant_peaks))]
    try:
        peak_time = [df.index[p].strftime('%H:%M') for p in significant_peaks]
    except:
        try:
            peak_time = [df.loc[df['hours']== p]['hours'] for p in significant_peaks]

        except Exception as e:
            print(f"Error formatting peak times: {e}")

    return {
        'peaks': significant_peaks.tolist(),
        'peak_times': peak_time,
        'peak_values': y[significant_peaks].tolist(),
        'C_base': C_base,
        'amplitudes': amplitudes.tolist(),
        'sigmas': sigmas.tolist(),
        'alphas': alphas.tolist()
    }


def print_pattern_summary(pattern, significant_peaks, C_base, sigmas, alphas):
    """
    Print summary of pattern processing.

    Parameters:
    - pattern: Name of the pattern.
    - significant_peaks: Array of peak indices.
    - C_base: Base consumption value.
    - sigmas: List of sigma values.
    - alphas: List of alpha values.
    """
    print(f"  Fitted {pattern} with {len(significant_peaks)} peaks")
    print(f"  Base consumption: {C_base:.2f}")
    print(f"  Sigmas (width): {', '.join([f'{s:.2f}' for s in sigmas])}")
    print(f"  Alphas (skew): {', '.join([f'{a:.2f}' for a in alphas])}")


def plot_actual_vs_predicted(df, results, hours, time_labels):
    """
    Plot actual vs predicted data for all patterns.

    Parameters:
    - df: DataFrame with time-indexed data.
    - results: Dictionary with model results.
    - hours: Array of hour indices.
    - time_labels: List of time labels for x-axis.
    """
    validation_metrics = []
    num_patterns = len(results)
    cols = min(2, num_patterns)
    rows = (num_patterns + cols - 1) // cols
    plt.figure(figsize=(14, 5 * rows))

    for i, pattern in enumerate(results.keys(), 1):
        plt.subplot(rows, cols, i)
        actual = df[pattern].values
        peaks = np.array(results[pattern]['peaks'])
        C_base = results[pattern]['C_base']
        amplitudes = results[pattern]['amplitudes']
        sigmas = results[pattern]['sigmas']
        alphas = results[pattern]['alphas']

        predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)
        real_integral = np.trapezoid(actual, dx=1)
        predicted_integral = np.trapezoid(predicted, dx=1)
        scale_factor = real_integral / predicted_integral
        predicted = predicted * scale_factor

        plt.plot(hours, actual, label='Actual', marker='o', color='blue')
        plt.plot(hours, predicted, label='Predicted', linestyle='--', color='red')
        plt.scatter(peaks, actual[peaks], color='green', s=100, zorder=5, label='Peaks')

        # plt.title(f"{pattern} - Actual vs Predicted")
        plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
        # plt.xlabel("Hour")
        # plt.ylabel(f"Flow (m³/h) {pattern}")
        plt.ylabel(f"{pattern}")
        plt.grid(True)
        plt.legend()

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mean_abs_error = np.mean(np.abs(actual - predicted))
        max_error = np.max(np.abs(actual - predicted))
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        print(f"  {pattern} - R²: {r2:.2f}")
        plt.annotate(f"RMSE: {rmse:.2f}\nMAE: {mean_abs_error:.2f}\nMax Error: {max_error:.2f}",
                     xy=(0.02, 0.96), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        print(f"  {pattern} - RMSE: {rmse:.2f}, MAE: {mean_abs_error:.2f}, Max Error: {max_error:.2f}")
        metric = {
            'pattern': pattern,
            'Mean flow': np.mean(actual),
            'RMSE': rmse,
            'RMSE as % of mean': (rmse / np.mean(actual)) * 100,
            'MAE': mean_abs_error,
            'MAE as % of mean': (mean_abs_error / np.mean(actual)) * 100,
            'Max Error': max_error,
            'R²': r2,
        }
        validation_metrics.append(metric)
    plt.tight_layout()
    # validation metric to frame and excel export
    validation_metrics_df = pd.DataFrame(validation_metrics)
    validation_metrics_df = validation_metrics_df.round(2)

    validation_metrics_df.to_excel('validation_metrics.xlsx', index=False)

def plot_model_components(df, results, hours, time_labels):
    """
    Plot model components for all patterns.

    Parameters:
    - df: DataFrame with time-indexed data.
    - results: Dictionary with model results.
    - hours: Array of hour indices.
    - time_labels: List of time labels for x-axis.
    """
    num_patterns = len(results)
    cols = min(2, num_patterns)
    rows = (num_patterns + cols - 1) // cols
    plt.figure(figsize=(14, 5 * rows))

    for i, pattern in enumerate(results.keys(), 1):
        plt.subplot(rows, cols, i)
        actual = df[pattern].values
        peaks = np.array(results[pattern]['peaks'])
        C_base = results[pattern]['C_base']
        amplitudes = results[pattern]['amplitudes']
        sigmas = results[pattern]['sigmas']
        alphas = results[pattern]['alphas']

        plt.plot(hours, actual, label='Actual', marker='o', color='blue')
        plt.axhline(y=C_base, color='gray', linestyle=':', label='$C_{base}$')

        for j, peak in enumerate(peaks):
            component = skewed_gaussian(hours, amplitudes[j], peak, sigmas[j], alphas[j])
            plt.plot(hours, C_base + component, linestyle='--', alpha=0.5,
                     label=f'Peak at {results[pattern]["peak_times"][j][:-3]}: ')

        predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)
        plt.plot(hours, predicted, label='Prediction', color='red', linewidth=2)
        plt.scatter(peaks, actual[peaks], color='green', s=100, zorder=5, label='Peaks')

        # plt.title(f"{pattern} - Model Components")
        plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
        # plt.xlabel("Hour")
        # plt.ylabel("Flow (m³/h)")
        # rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        # mean_abs_error = np.mean(np.abs(actual - predicted))
        # max_error = np.max(np.abs(actual - predicted))
        # plt.annotate(f"RMSE: {rmse:.2f}\nMAE: {mean_abs_error:.2f}\nMax Error: {max_error:.2f}",
        #              xy=(0.02, 0.8), xycoords='axes fraction',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.ylabel(f"{pattern}")
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()


def print_model_parameters(results):
    """
    Print model parameters for all patterns.

    Parameters:
    - results: Dictionary with model results.
    """
    print("\nModel Parameters:")
    for pattern, params in results.items():
        print(f"\n{pattern}:")
        print(f"  Base consumption: {params['C_base']:.2f}")
        print("  Peaks:")
        for i, peak in enumerate(params['peaks']):
            print(f"    Peak at {params['peak_times'][i]}: "
                  f"Value={params['peak_values'][i]:.2f}, "
                  f"Width(σ)={params['sigmas'][i]:.2f}, "
                  f"Skew(α)={params['alphas'][i]:.2f}")


def save_model_parameters(results):
    """
    Save model parameters for all patterns.

    Parameters:
    - results: Dictionary with model results.
    """
    print("\nModel Parameters:")
    all_alphas_for_all_patterns_for_hist = []
    models_parameters = []
    for pattern, params in results.items():
        # add the pattern name and all min max range of all parameters in the results dict, make sure to round each float element to 2 decimal places
        range_parameters = {
            'pattern': pattern,
            'N peaks': len(params['peaks']),
            'C_base': params['C_base'],
            'Flow Range': (float(f'{min(params['peak_values']):.2f}'), float(f'{max(params['peak_values']):.2f}')),
            # 'Integral Actual': float(f'{np.trapezoid(params['peak_values'], dx=1):.2f}'),
            'Amplitude Range': (float(f'{min(params['amplitudes']):.2f}'), float(f'{max(params['amplitudes']):.2f}')),
            'Sigma Range': (float(f'{min(params['sigmas']):.2f}'), float(f'{max(params['sigmas']):.2f}')),
            'Alpha Range': (float(f'{min(params['alphas']):.2f}'), float(f'{max(params['alphas']):.2f}')),
        }
        all_alphas_for_all_patterns_for_hist.extend(params['alphas'])

        models_parameters.append(range_parameters)
    models_parameters_df = pd.DataFrame(models_parameters)
    # if any value in models_parameters_df is float - round to 2 decimal places
    models_parameters_df = models_parameters_df.round(2)
    models_parameters_df.to_excel('model_parameters.xlsx', index=False)
    # Save histogram of alphas
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_alphas_for_all_patterns_for_hist, bins=20, color='blue', alpha=0.7)
    # plt.title('Histogram of Alphas for All Patterns')
    # plt.xlabel('Alpha')
    # plt.ylabel('Frequency, Alphas for All Patterns')
    # plt.grid(True)
    # plt.show()


def decompose_patterns(df, patterns=None, max_peaks=DefaultsParameters.MAX_PEAKS, hours=np.arange(24)):
    """
    Decompose patterns from a DataFrame, detecting peaks and fitting a model.

    Parameters:
    - df: DataFrame with time-indexed data and pattern columns.
    - patterns: List of column names to process (default: all columns).
    - max_peaks: Maximum number of peaks to detect (default: 6).
    - hours: Array of hour indices for plotting (default: 0 to 23).

    Returns:
    - results: Dictionary with model parameters and peak information for each pattern.
    """
    if patterns is None:
        patterns = df.columns

    results = {}

    for pattern in patterns:
        print(f"\nProcessing {pattern}...")
        y = df[pattern].values
        x = np.arange(len(y))

        # Detect peaks
        significant_peaks = find_significant_peaks(df[[pattern]])

        print(f"  Detected {len(significant_peaks)} significant peaks")

        # Fit model
        C_base, amplitudes, sigmas, alphas = fit_pattern(x, y, significant_peaks, pattern)

        # Store results
        results[pattern] = store_results(df, y, significant_peaks, C_base, amplitudes, sigmas, alphas)

        # Print summary
        print_pattern_summary(pattern, significant_peaks, C_base, sigmas, alphas)

    return results


def plot_pattern(df, patterns=None, max_peaks=DefaultsParameters.MAX_PEAKS, hours=np.arange(24)):
    """
    Process and plot patterns from a DataFrame, detecting peaks and fitting a model.

    Parameters:
    - df: DataFrame with time-indexed data and pattern columns.
    - patterns: List of column names to process (default: all columns).
    - max_peaks: Maximum number of peaks to detect (default: 6).
    - hours: Array of hour indices for plotting (default: 0 to 23).

    Returns:
    - results: Dictionary with model parameters and peak information for each pattern.
    """
    if patterns is None:
        patterns = df.columns

    results = {}
    # time_labels = [t.strftime('%H:%M') for t in df.index]
    time_labels = [t.strftime('%H') for t in df.index]

    #
    # for pattern in patterns:
    #     print(f"\nProcessing {pattern}...")
    #     y = df[pattern].values
    #     x = np.arange(len(y))
    #
    #     # Detect peaks
    #     significant_peaks = find_significant_peaks(df[[pattern]])
    #
    #     print(f"  Detected {len(significant_peaks)} significant peaks")
    #
    #     # Fit model
    #     C_base, amplitudes, sigmas, alphas = fit_pattern(x, y, significant_peaks)
    #
    #     # Store results
    #     results[pattern] = store_results(df, y, significant_peaks, C_base, amplitudes, sigmas, alphas)
    #
    #     # Print summary
    #     print_pattern_summary(pattern, significant_peaks, C_base, sigmas, alphas)
    results = decompose_patterns(df, patterns, max_peaks, hours)
    # Plot results
    # plot_actual_vs_predicted(df, results, hours, time_labels)
    plot_model_components(df, results, hours, time_labels)

    # Print model parameters
    print_model_parameters(results)
    save_model_parameters(results)

    plt.show()
    return results


if __name__ == "__main__":

    # --- Load Data ---

    df = pd.read_excel('demand_pattern.xlsx')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df.set_index('Time', inplace=True)
    # Call the main function
    # print(df.columns)
    # exit()
    # patterns = ['Flow D1 (m³/h)', 'Flow D2 (m³/h)', 'Flow D3 (m³/h)',
    #        'Flow Abu-Bakar2021EP (m³/h)', 'Flow Abu-Bakar2021LM (m³/h)',
    #        'Flow Abu-Bakar2021EM (m³/h)', 'Flow Abu-Bakar2021MP (m³/h)',
    #        ['Flow D1 (m³/h)', 'Flow D2 (m³/h)', 'Flow D3 (m³/h)',
    #        'Flow Abu-Bakar2021EP (m³/h)', 'Flow Abu-Bakar2021LM (m³/h)',
    #        'Flow Abu-Bakar2021EM (m³/h)', 'Flow Abu-Bakar2021MP (m³/h)',
    #        'Flow Plantak2022a (m³/h)', 'Flow Plantak2022b (m³/h)',
    #        'Flow Mauro2021a (m³/h)', 'Flow Mauro2021b (m³/h)',
    #        'Flow Nemati2023a (Gallons/h)', 'Flow Nemati2023b (Gallons/h)']
    patterns = [
       'Abu-Bakar2021EP (m³/h)', 'Abu-Bakar2021MP (m³/h)',
       'Plantak2022a (m³/h)', 'Plantak2022b (m³/h)',
       'Nemati2023a (Gallons/h)', 'Nemati2023b (Gallons/h)']
    # results = plot_pattern(df, patterns=patterns)
    results = plot_pattern(df, df.columns)









