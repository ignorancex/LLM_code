from root.correlations.correlations import Correlations
from root.data.data import Data
import mlx.core as mx

from root.plotter.plotter import Plotter


def create_decreasingly_correlated_data(num_datasets, num_patterns, num_features):
    # eps should be 50 interfvals between 0 and 1
    datasets = []
    for i in range(num_datasets):
        eps = 1 - ((i + 1) / num_datasets)  # create evenly spaced data
        patterns = Data.generate_correlated_patterns(
            num_patterns, num_features, eps=eps
        )
        datasets.append(patterns)
    return mx.array(datasets)


def create_uncorrelated_data(num_datasets, num_patterns, num_features):
    datasets = []
    for _ in range(num_datasets):
        patterns = Data.generate_rademacher_patterns(num_patterns, num_features)
        datasets.append(patterns)
    return mx.array(datasets)


def create_simulated_real_world_data(num_datasets, num_patterns, num_features):
    datasets = []
    patterns = Data.generate_real_world_patterns(num_patterns, num_features)
    datasets.append(patterns)
    return mx.array(datasets)


def create_mnist_subset():
    datasets = []
    for i in range(165, 200, 3):
        data = Data.create_subset_of_mnist_with_goal_hd(50, float(i))
        print(i, Correlations.calc_average_hd(data))
        if data.shape[0] == 50:
            datasets.append(data)

    data = mx.array(datasets).astype(mx.float32)
    Data.save("mnist_subsets_all_165_190_3", data)


def main():
    # data = create_decreasingly_correlated_data(50, 5000, 64)
    # Data.save("5000_corr_datasets", data)

    # data = create_simulated_real_world_data(1, 500, 64)
    # Data.save("real_world_sim_data", data)

    create_mnist_subset()


if __name__ == "__main__":
    main()
