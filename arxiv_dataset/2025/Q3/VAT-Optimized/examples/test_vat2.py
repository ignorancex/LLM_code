from vat_clustering.vat import vat, plot_vat
from vat_clustering.optimization import vat_optimized
from vat_clustering.vat_cython import vat_cython
import numpy as np
import time

def test_vat_execution(R, dataset_name):
    print(f"\n🔹 Testing {dataset_name} Dataset...")

    # Standard VAT
    start_time = time.time()
    RV, order = vat(R)
    vat_time = time.time() - start_time
    print(f"⏱ Standard VAT Execution Time: {vat_time:.4f} seconds")

    # Optimized VAT (Numba)
    start_time = time.time()
    RV_opt, order_opt = vat_optimized(R)
    opt_vat_time = time.time() - start_time
    print(f"⚡ Optimized VAT Execution Time (Numba): {opt_vat_time:.4f} seconds")

    # Cython VAT
    start_time = time.time()
    RV_cython, order_cython = vat_cython(R)
    cython_vat_time = time.time() - start_time
    print(f"🚀 Cython VAT Execution Time: {cython_vat_time:.4f} seconds")

    # Speedup Comparison
    print(f"🔍 Speedup (Cython vs. Python): {vat_time / cython_vat_time:.2f}x faster")
    print(f"🔍 Speedup (Numba vs. Python): {vat_time / opt_vat_time:.2f}x faster\n")

# Load datasets (Ensure paths are correct)
datasets = {
    "Iris": np.load("data/iris_dissimilarity.npy"),
    "Spotify (500x500)": np.load("data/spotify_dissimilarity_subset.npy"),
    "Blobs": np.load("data/blobs_dissimilarity.npy"),
    "Circles": np.load("data/circles_dissimilarity.npy"),
    "GMM": np.load("data/gmm_dissimilarity.npy"),
    "Mall Customers": np.load("data/mall_dissimilarity.npy"),
    "Moons": np.load("data/moons_dissimilarity.npy"),
}

# Run tests on all datasets
for name, R in datasets.items():
    test_vat_execution(R, name)
