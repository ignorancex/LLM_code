import argparse
import os

import numpy as np
import scipy.io as sio
from tqdm import tqdm

SUPPORTED_DATASETS = ["darcy", "ns-bounded", "ns-nonbounded", "burgers", "helmholtz", "poisson"]

STATS = {
    "darcy": {"mean": [7.5, 5.69201936e-03], "std": [4.5, 3.79030361e-03], "min": [3.0, -0.28737752], "max": [12.0, 0.11770357]},
    'ns-bounded': {'mean': [1.80715032, 2.87310427], 'std': [1.00997056, 1.72194188], 'min': [0.0, 0.0], 'max': [10.00000151, 10.00000203]},
    "ns-nonbounded": {"mean": [0, 0], "std": [0.26211293, 0.2561266], "min": [-1.3913461, -1.31577837], "max": [1.58353412, 1.4743135]},
    "burgers": {"mean": [0, 0], "std": [0.27366568, 0.20088117], "min": [-1.28759708, -1.28759708], "max": [1.21768836, 1.21768836]},
    "helmholtz": {"mean": [0, 1.05050595e-05], "std": [0.2844538, 0.00428004], "min": [-2.09217139, -0.02734944], "max": [2.13316352, 0.02774104]},
    "poisson": {"mean": [0, 9.67226952e-06], "std": [0.2919494, 0.00417478], "min": [-2.13132663, -0.02657227], "max": [2.16163561, 0.02717429]},
    "__version__": "2.0",
}


def load_darcy(file_path):
    """Load Darcy flow dataset."""
    data = sio.loadmat(file_path)
    a = data["thresh_a_data"]
    u = data["thresh_p_data"]
    return a, u


def load_ns_bounded(file_path):
    """Load NS-bounded dataset."""
    data = np.load(file_path)
    a = data[..., 4]
    u = data[..., 8]
    return a, u


def load_ns_nonbounded(file_path):
    """Load NS-nonbounded dataset."""
    data = sio.loadmat(file_path)
    a = data["a"]
    u = data["u"][..., -1]  # taking last timestep
    return a, u


def load_burgers(file_path):
    """Load Burgers dataset."""
    data = sio.loadmat(file_path)
    a = data["input"]
    u = data["output"]
    # Repeat the first array to match the shape of the second array
    a = np.repeat(a[..., np.newaxis], u.shape[-1], axis=-1)
    return a, u


def load_helmholtz(file_path):
    """Load Helmholtz dataset."""
    data = sio.loadmat(file_path)
    a = data["f_data"]
    u = data["psi_data"]
    return a, u


def load_poisson(file_path):
    """Load Poisson dataset."""
    data = sio.loadmat(file_path)
    a = data["f_data"]
    u = data["phi_data"]
    return a, u


def get_dataset_info(dataset_name, training=True):
    """Get dataset-specific information."""
    training_set_info = {
        "darcy": {
            "loader": load_darcy,
            "path_pattern": "data/training/darcy/darcy_{}.mat",
            "range": range(1, 6),
        },
        "ns-bounded": {
            "loader": load_ns_bounded,
            "path_pattern": "data/training/ns-bounded/{}/v.npy",
            "range": [1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42],
        },
        "ns-nonbounded": {
            "loader": load_ns_nonbounded,
            "path_pattern": "data/training/ns-nonbounded/ns-nonbounded_{}.mat",
            "range": range(1, 51),
        },
        "burgers": {
            "loader": load_burgers,
            "path_pattern": "data/training/burger/burger_{}.mat",
            "range": range(1, 6),
        },
        "helmholtz": {
            "loader": load_helmholtz,
            "path_pattern": "data/training/helmholtz/helmholtz_{}.mat",
            "range": range(1, 6),
        },
        "poisson": {
            "loader": load_poisson,
            "path_pattern": "data/training/poisson/poisson_{}.mat",
            "range": range(1, 6),
        },
    }
    test_set_info = {
        "darcy": {
            "loader": load_darcy,
            "path_pattern": "data/testing/darcy.mat",
            "range": range(1),
        },
        "ns-bounded": {
            "loader": load_ns_bounded,
            "path_pattern": "data/testing/ns-bounded/0/v.npy",
            "range": range(1),
        },
        "ns-nonbounded": {
            "loader": load_ns_nonbounded,
            "path_pattern": "data/testing/ns-nonbounded.mat",
            "range": range(1),
        },
        "burgers": {
            "loader": load_burgers,
            "path_pattern": "data/testing/burgers.mat",
            "range": range(1),
        },
        "helmholtz": {
            "loader": load_helmholtz,
            "path_pattern": "data/testing/helmholtz.mat",
            "range": range(1),
        },
        "poisson": {
            "loader": load_poisson,
            "path_pattern": "data/testing/poisson.mat",
            "range": range(1),
        },
    }

    info = training_set_info if training else test_set_info
    return info[dataset_name]


def calculate_dataset_statistics(dataset_name):
    """Calculate mean and standard deviation for each channel in the raw dataset."""
    info = get_dataset_info(dataset_name)
    loader = info["loader"]
    path_pattern = info["path_pattern"]
    file_range = info["range"]

    # Initialize variables for online statistics calculation
    counts = np.zeros(2)
    mean = np.zeros(2)
    M2 = np.zeros(2)

    # Track min and max values
    min_vals = np.array([np.inf, np.inf])
    max_vals = np.array([-np.inf, -np.inf])

    print(f"\nProcessing raw {dataset_name} dataset...")

    # Process each file
    for j in tqdm(file_range):
        file_path = path_pattern.format(j)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        # Load raw data
        try:
            a, u = loader(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

        # Update statistics for both channels
        for i, data in enumerate([a, u]):
            data_flat = data.reshape(-1)

            # Update min/max
            min_vals[i] = min(min_vals[i], np.min(data_flat))
            max_vals[i] = max(max_vals[i], np.max(data_flat))

            # Update mean and variance
            delta = data_flat - mean[i]
            counts[i] += len(data_flat)
            mean[i] += np.sum(delta) / counts[i]
            delta2 = data_flat - mean[i]
            M2[i] += np.sum(delta * delta2)

    # Calculate final statistics
    variance = M2 / (counts - 1)
    std = np.sqrt(variance)

    print(f"\nDataset: {dataset_name}")
    for i in range(2):
        print(f"Channel {i + 1}:")
        print(f"  Mean: {mean[i]:.6f}")
        print(f"  Std: {std[i]:.6f}")
        print(f"  Min: {min_vals[i]:.6f}")
        print(f"  Max: {max_vals[i]:.6f}")

    stats = {"mean": mean, "std": std, "min": min_vals, "max": max_vals}

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate statistics for raw PDE datasets.")
    parser.add_argument("dataset", type=str, help=f"Name of the dataset ({', '.join(SUPPORTED_DATASETS)}) or 'all' to process all datasets")
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        stats = {}
        for dataset in SUPPORTED_DATASETS:
            stats[dataset] = calculate_dataset_statistics(dataset)
    else:
        if args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {args.dataset} not supported. Supported datasets are: {', '.join(SUPPORTED_DATASETS)}")
        stats = {args.dataset: calculate_dataset_statistics(args.dataset)}

    print(stats)
