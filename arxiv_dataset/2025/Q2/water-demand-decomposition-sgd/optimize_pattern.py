import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.special import erf



def process_sequences(arr):
    if len(arr) == 0:
        return np.array([])

    result = []
    i = 0
    while i < len(arr):
        # Find the start of a sequence
        start = i
        # Continue while numbers are consecutive
        while i + 1 < len(arr) and arr[i + 1] == arr[i] + 1:
            i += 1
        # Include the current number in the sequence
        end = i

        # Process the sequence
        seq_len = end - start + 1
        if seq_len == 1:
            # Single number, keep it
            result.append(arr[start])
        elif seq_len == 2:
            # Two consecutive numbers, keep the first
            result.append(arr[start])
        else:
            # Three or more numbers
            # If odd length, take middle; if even, take center-left
            mid_idx = start + (seq_len // 2)
            if seq_len % 2 == 1:
                # Odd length, take middle
                result.append(arr[mid_idx])
            else:
                # Even length, take center-left
                result.append(arr[mid_idx - 1])

        i += 1

    return np.array(result)


# --- Define Skewed Gaussian Function ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    """Skewed Gaussian function."""
    norm_gaussian = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew


# --- Load Data ---
try:
    df = pd.read_excel('demand_pattern.xlsx')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df.set_index('Time', inplace=True)
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data for testing if file not found
    hours = [f"{h:02d}:00:00" for h in range(24)]
    sample_data = {
        'Pattern1': 100 + 50 * np.sin(np.linspace(0, 2 * np.pi, 24)),
        'Pattern2': 75 + 45 * np.cos(np.linspace(0, 2 * np.pi, 24))
    }
    df = pd.DataFrame(sample_data, index=hours)
    df.index = pd.to_datetime(df.index, format='%H:%M:%S').time
    print("Using sample data for demonstration")


# --- Function to create the composite model ---
def composite_model(x, peaks, C_base, amplitudes, sigmas, alphas):
    """Composite model of multiple skewed Gaussians with fixed peaks."""
    result = np.full(len(x), C_base)
    for i, peak in enumerate(peaks):
        component = skewed_gaussian(x, amplitudes[i], peak, sigmas[i], alphas[i])
        result += component
    return result


# --- Function to fit model ensuring it passes through peaks ---
def fit_with_peak_constraints(x, y, peaks):
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
        print(f"Warning: Constrained optimization failed. Trying unconstrained...")
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


# --- Process each pattern ---
results = {}

for pattern in df.columns:
    print(f"\nProcessing {pattern}...")
    y = df[pattern].values
    x = np.arange(len(y))

    # Find peaks
    dy = np.diff(np.concatenate([[y[0]], y, [y[-1]]]))
    peaks = np.where((dy[:-2] > 0) & (dy[1:-1] <= 0))[0]  # where derivative changes from + to -

    plateaus = np.where(dy[:-1] == 0)[0] + 1

    plateaus = process_sequences(plateaus)

    peaks = np.concatenate((peaks, plateaus))


    # Add endpoints if they're local maxima
    if y[0] > y[1]:
        peaks = np.append([0], peaks)
    if y[-1] > y[-2]:
        peaks = np.append(peaks, [len(y) - 1])

    peaks = np.unique(peaks)
    peaks = peaks[peaks <= len(y) - 1]  # Ensure peaks are within bounds
    # Filter peaks to keep only significant ones
    peak_heights = y[peaks]
    # threshold = np.mean(y) + 0.25 * (np.max(y) - np.mean(y))  # Dynamic threshold
    # significant_peaks = peaks[peak_heights > threshold]
    significant_peaks = peaks
    # Ensure we have at least 2 peaks for a good fit
    if len(significant_peaks) < 2:
        # Add next highest point
        non_peak_indices = np.setdiff1d(np.arange(len(y)), significant_peaks)
        non_peak_values = y[non_peak_indices]
        additional_indices = non_peak_indices[np.argsort(non_peak_values)[-2:]]
        significant_peaks = np.unique(np.concatenate([significant_peaks, additional_indices]))

    # Cap at maximum 6 peaks to avoid overfitting
    if len(significant_peaks) > 6:
        peak_values = y[significant_peaks]
        top_indices = np.argsort(peak_values)[-6:]
        significant_peaks = significant_peaks[top_indices]

    # Sort peaks by position
    significant_peaks = np.sort(significant_peaks)

    print(f"  Detected {len(significant_peaks)} significant peaks")

    # Fit the model ensuring it passes through peaks
    C_base, amplitudes, sigmas, alphas = fit_with_peak_constraints(x, y, significant_peaks)

    # Store results
    results[pattern] = {
        'peaks': significant_peaks.tolist(),
        'peak_times': [df.index[p].strftime('%H:%M') for p in significant_peaks],
        'peak_values': y[significant_peaks].tolist(),
        'C_base': C_base,
        'amplitudes': amplitudes.tolist(),
        'sigmas': sigmas.tolist(),
        'alphas': alphas.tolist()
    }

    print(f"  Fitted {pattern} with {len(significant_peaks)} peaks")
    print(f"  Base consumption: {C_base:.2f}")
    print(f"  Sigmas (width): {', '.join([f'{s:.2f}' for s in sigmas])}")
    print(f"  Alphas (skew): {', '.join([f'{a:.2f}' for a in alphas])}")

# --- Plot Actual vs Predicted ---
hours = np.arange(24)
time_labels = [t.strftime('%H:%M') for t in df.index]

num_patterns = len(results)
cols = min(2, num_patterns)
rows = (num_patterns + cols - 1) // cols
plt.figure(figsize=(14, 5 * rows))

for i, pattern in enumerate(results.keys(), 1):
    plt.subplot(rows, cols, i)

    # Get actual data
    actual = df[pattern].values

    # Get extracted parameters
    peaks = np.array(results[pattern]['peaks'])
    C_base = results[pattern]['C_base']
    amplitudes = results[pattern]['amplitudes']
    sigmas = results[pattern]['sigmas']
    alphas = results[pattern]['alphas']

    # Generate predicted values
    predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)

    # --- Force area (integral) to match ---
    real_integral = np.trapezoid(actual, dx=1)
    predicted_integral = np.trapezoid(predicted, dx=1)
    scale_factor = real_integral / predicted_integral
    predicted = predicted * scale_factor

    # Plot
    plt.plot(hours, actual, label='Actual', marker='o', color='blue')
    plt.plot(hours, predicted, label='Predicted', linestyle='--', color='red')

    # Mark peaks
    plt.scatter(peaks, actual[peaks], color='green', s=100, zorder=5, label='Peaks')

    plt.title(f"{pattern} - Actual vs Predicted")
    plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
    plt.xlabel("Hour")
    plt.ylabel("Flow (m³/h)")
    plt.grid(True)
    plt.legend()

    # Calculate and display error metrics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mean_abs_error = np.mean(np.abs(actual - predicted))
    max_error = np.max(np.abs(actual - predicted))

    plt.annotate(f"RMSE: {rmse:.2f}\nMAE: {mean_abs_error:.2f}\nMax Error: {max_error:.2f}",
                 xy=(0.02, 0.96), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()

# --- Additional Diagnostic Plot (Components) ---
plt.figure(figsize=(14, 5 * rows))

for i, pattern in enumerate(results.keys(), 1):
    plt.subplot(rows, cols, i)

    # Get actual data
    actual = df[pattern].values

    # Get extracted parameters
    peaks = np.array(results[pattern]['peaks'])
    C_base = results[pattern]['C_base']
    amplitudes = results[pattern]['amplitudes']
    sigmas = results[pattern]['sigmas']
    alphas = results[pattern]['alphas']

    # Plot actual data
    plt.plot(hours, actual, label='Actual', marker='o', color='blue')

    # Plot base consumption
    plt.axhline(y=C_base, color='gray', linestyle=':', label='Base Consumption')

    # Plot each component
    for j, peak in enumerate(peaks):
        component = skewed_gaussian(hours, amplitudes[j], peak, sigmas[j], alphas[j])
        plt.plot(hours, C_base + component, linestyle='--', alpha=0.5,
                 label=f'Peak at {results[pattern]["peak_times"][j]}')

    # Plot total prediction
    predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)
    plt.plot(hours, predicted, label='Total Prediction', color='red', linewidth=2)

    plt.title(f"{pattern} - Model Components")
    plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
    plt.xlabel("Hour")
    plt.ylabel("Flow (m³/h)")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# --- Print model parameters ---
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