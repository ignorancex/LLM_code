import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw

# Generate synthetic time-series data
time_series = np.sin(np.linspace(0, 2 * np.pi, 100)).flatten()  # Flatten to 1-D
reference_series = np.sin(np.linspace(0, 2 * np.pi, 120) + 0.5).flatten()  # Flatten to 1-D

# Ensure both time_series and reference_series are 1-D
print(f"Time series shape: {time_series.shape}, Reference series shape: {reference_series.shape}")

# Define custom distance function for scalar values
def scalar_distance(u, v):
    return abs(u - v)

# Apply DTW to align the sequences using the custom distance function
distance, path = fastdtw(time_series, reference_series, dist=scalar_distance)

# Plot the aligned sequences and highlight the warping path
plt.figure(figsize=(10, 5))
plt.plot(time_series, label="Time Series")
plt.plot(
    np.interp(np.linspace(0, 100, 120), np.arange(120), reference_series),
    label="Reference Series",
    alpha=0.7
)

# Highlight the warping path
for (i, j) in path:
    plt.plot([i, j * 100 / 120], [time_series[i], reference_series[j]], color='gray', alpha=0.5)

plt.title(f"DTW Alignment (Distance: {distance:.2f})")
plt.xlabel("Time Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
