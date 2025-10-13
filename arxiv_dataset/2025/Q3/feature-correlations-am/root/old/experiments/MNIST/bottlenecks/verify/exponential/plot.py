import numpy as np

from root.correlation.metrics.distance import AverageDCor
from root.correlation.metrics.hamming import (
    BinaryMeanPairwiseHammingDistance,
    BinaryMinPairwiseHammingDistance,
)
from root.experiments.MNIST.bottlenecks.verify.exponential.test_subsets import (
    get_mean_hds,
    get_min_hds,
    load_subsets,
)
from root.experiments.MNIST.bottlenecks.verify.plot import plot_scatter_with_fitted_line


def main():
    # create_subsets() #do when needed
    subsets = load_subsets("all_subsets.npy")
    memcaps = load_subsets("memcaps.npy")
    print(memcaps)
    hds = get_mean_hds(subsets)
    min_hds = []
    mean_hds = []
    distcorrs = []
    for sets in subsets:
        min_hd = BinaryMinPairwiseHammingDistance.calculate(sets)
        mean_hd = BinaryMeanPairwiseHammingDistance.calculate(sets)
        distcorr = AverageDCor.calculate(sets)
        min_hds.append(min_hd)
        mean_hds.append(mean_hd)
        distcorrs.append(distcorr)

    print(min_hds)
    print(mean_hds)
    print(distcorrs)

    # dataset = np.vstack((hds, memcaps))
    dataset1 = np.vstack((min_hds, memcaps))
    dataset2 = np.vstack((mean_hds, memcaps))
    dataset3 = np.vstack((distcorrs, memcaps))

    plot_scatter_with_fitted_line(dataset1, "memcaps.png", "Mean HD", "Mem Cap", "red")
    plot_scatter_with_fitted_line(
        dataset2, "memcaps.png", "Mean HD", "Mem Cap", "orange"
    )
    plot_scatter_with_fitted_line(
        dataset3, "memcaps.png", "Mean HD", "Mem Cap", "blue", no_xlim=True
    )


if __name__ == "__main__":
    main()
