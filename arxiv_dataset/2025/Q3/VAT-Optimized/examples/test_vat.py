import numpy as np
import time
from vat_clustering.vat import vat, plot_vat
from vat_clustering.optimization import vat_optimized

# Function to test VAT performance
def test_vat_execution(R, dataset_name):
    print(f"🔹 Testing {dataset_name} Dataset...")

    # Measure execution time for standard VAT
    start_time = time.time()
    RV, order = vat(R)
    vat_time = time.time() - start_time
    print(f"⏱ Standard VAT Execution Time: {vat_time:.4f} seconds")

    # Measure execution time for optimized VAT
    start_time = time.time()
    RV_opt, order_opt = vat_optimized(R)
    opt_vat_time = time.time() - start_time
    print(f"⚡ Optimized VAT Execution Time: {opt_vat_time:.4f} seconds")

    # Compare speedup
    speedup = vat_time / opt_vat_time
    print(f"🚀 Speed Improvement: {speedup:.2f}x faster\n")

    # Plot VAT Results (Optional)
    plot_vat(RV, title=f"VAT Reordered Image - {dataset_name}")
    plot_vat(RV_opt, title=f"Optimized VAT Reordered Image - {dataset_name}")

# Load datasets (Adjust paths if needed)
datasets = {
    "Iris": np.load("data/iris_dissimilarity.npy"),
    "Mall Customers": np.load("data/mall_dissimilarity.npy"),
    "Spotify (500x500)": np.load("data/spotify_dissimilarity_subset.npy"),
    "Blobs": np.load("data/blobs_dissimilarity.npy"),
    "Moons": np.load("data/moons_dissimilarity.npy"),
    "Circles": np.load("data/circles_dissimilarity.npy"),
    "Gaussian Mixture": np.load("data/gmm_dissimilarity.npy")
}

# Run tests on all datasets
for name, R in datasets.items():
    test_vat_execution(R, name)
