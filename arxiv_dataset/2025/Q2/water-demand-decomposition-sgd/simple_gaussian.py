import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import erf

# --- Define Skewed Gaussian Function ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    """Skewed Gaussian function."""
    norm_gaussian = np.exp(-(x - mu)**2 / (2 * sigma**2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew

# --- Load Data ---
df = pd.read_excel('demand_pattern.xlsx')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
df.set_index('Time', inplace=True)

# --- Detect Peaks and Fit ---
results = {}
sigmas = {}
alphas = {}

for pattern in df.columns:
    y = df[pattern].values
    x = np.arange(len(y))
    pattern_sigmas = []
    pattern_alphas = []

    # Find peaks
    dy = np.diff(y)
    peaks = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1  # where derivative changes from + to -
    plato = np.where(dy[:-1] == 0)[0] + 1
    peaks = list(peaks) + list(plato)
    peaks = np.unique(peaks)
    peaks = np.sort(peaks)

    peak_amps = y[peaks]
    peak_times = df.index[peaks]

    C_base = np.min(y)  # Base consumption

    results[pattern] = {
        'A': peak_amps.tolist(),
        't': [t.hour for t in peak_times],
        'C_base': C_base
    }

    # Fit around each peak
    for peak in peaks:
        window_size = 5  # +/- 2 hours
        start = max(0, peak - window_size)
        end = min(len(y), peak + window_size)

        x_window = x[start:end]
        y_window = y[start:end] - np.min(y[start:end])  # baseline correction

        try:
            # Initial guess: A, mu (fixed), sigma, alpha
            p0 = [y[peak], peak, 2, 0]
            bounds = ([0, peak-0.01, 0.5, -5], [np.inf, peak+0.01, 10, 5])  # tight bounds on mu
            popt, _ = curve_fit(skewed_gaussian, x_window, y_window, p0=p0, bounds=bounds)
            fitted_sigma = abs(popt[2])
            fitted_alpha = popt[3]
            pattern_sigmas.append(fitted_sigma)
            pattern_alphas.append(fitted_alpha)
        except Exception as e:
            print(f"Fit failed at peak {peak} in pattern {pattern}: {e}")
            pattern_sigmas.append(np.nan)
            pattern_alphas.append(np.nan)

    sigmas[pattern] = pattern_sigmas
    alphas[pattern] = pattern_alphas

# --- Plot Actual vs Predicted ---
import math

hours = np.arange(24)
time_labels = [str(t) for t in df.index]

num_patterns = len(results)
cols = 2
rows = math.ceil(num_patterns / cols)
plt.figure(figsize=(14, 4 * rows))

for i, pattern in enumerate(results.keys(), 1):
    plt.subplot(rows, cols, i)

    # Get actual data
    actual = df[pattern].values

    # Get extracted parameters
    C_base = results[pattern]['C_base']
    A_peaks = results[pattern]['A']
    t_peaks = results[pattern]['t']
    sigmas_ = sigmas[pattern]
    alphas_ = alphas[pattern]

    # Predict model
    def model_sum(t):
        total = C_base
        for A, t0, sigma, alpha in zip(A_peaks, t_peaks, sigmas_, alphas_):
            if np.isnan(sigma) or np.isnan(alpha):
                continue
            total += skewed_gaussian(t, A, t0, sigma, alpha)
        return total

    predicted = [model_sum(t) for t in hours]

    # --- Force area (integral) to match ---
    real_integral = np.trapezoid(actual, dx=1)
    predicted_integral = np.trapezoid(predicted, dx=1)
    scale_factor = real_integral / predicted_integral
    predicted = np.array(predicted) * scale_factor

    # Plot
    plt.plot(hours, actual, label='Actual', marker='o')
    plt.plot(hours, predicted, label='Predicted', linestyle='--')
    plt.title(f"{pattern} - Actual vs Predicted")
    plt.xticks(hours, time_labels, rotation=90)
    plt.xlabel("Hour")
    plt.ylabel("Flow (m³/h)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
