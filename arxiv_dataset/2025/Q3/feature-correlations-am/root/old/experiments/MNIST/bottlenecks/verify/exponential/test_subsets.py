import os

import numpy as np
from root.experiments.MNIST.bottlenecks.verify.plot import (
    make_sorted_set,
    plot_scatter_with_fitted_line,
)
from root.experiments.MNIST.information.hamming import (
    get_bottleneck_idxs,
    mean_hamming_distance,
    min_hamming_distance,
)

from root.utils import record_stdout
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity


def get_min_hds(subsets):
    min_hds = []
    for subset in subsets:
        min_hds.append(min_hamming_distance(subset))
    return min_hds


def get_mean_hds(subsets):
    mean_hds = []
    for subset in subsets:
        mean_hds.append(mean_hamming_distance(subset))
    return mean_hds


def load_subsets(filepath):
    script_dir = os.path.dirname(__file__)
    abs_path = os.path.join(script_dir, filepath)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File at {filepath} does not exist.")

    array = []
    with open(abs_path, "rb") as f:
        array = np.load(f)
    return array


def test_subsets(subsets):
    mem_caps = []
    for subset in subsets:
        bottleneck_idxs = get_bottleneck_idxs(subset)[0]
        mem_cap = get_memorization_capacity(
            subset,
            recall_quality=1.0,
            startAt=6,
            verbose=True,
            test_idxs=bottleneck_idxs,
        )
        mem_caps.append(mem_cap)

    return mem_caps


def create_subsets():
    subset1 = load_subsets("subsets_5_38_ones.npy")
    subset2 = load_subsets("subsets_38_120_one_twos.npy")
    subset3 = load_subsets("subsets_120_380_twos_random.npy")
    subsets = np.vstack((subset1, subset2, subset3))[0:-9]
    print(subsets.shape)  # (many, 50, 784)
    with open("all_subsets.npy", "wb") as f:
        np.save(f, subsets)


def main():
    # create_subsets() #do when needed
    subsets = load_subsets("all_subsets.npy")
    memcaps = test_subsets(subsets)
    with open("memcaps.npy", "wb") as f:
        np.save(f, memcaps)


if __name__ == "__main__":
    main()
