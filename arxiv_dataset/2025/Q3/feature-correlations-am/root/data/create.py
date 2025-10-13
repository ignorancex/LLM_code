import mlx.core as mx

from root.correlations.correlations import Correlations
from root.data.data import Data
from root.sims.plot import plot_with_config, prepare_plot_data


def plot_mnist():
    data = Data.load("all_subsets")
    data = data[:41, :, :]
    for i in range(data.shape[0]):
        print(Correlations.calc_average_hd(data[i, :, :]))

    plot_inputs = prepare_plot_data(
        "all_subsets",
        [
            ("k_max_subsets_at_n4", "n=4"),
            ("k_max_subsets_at_n5", "n=5"),
            ("k_max_subsets_at_n6", "n=6"),
            ("k_max_subsets_at_n7", "n=7"),
        ],
        threshold=50,
        truncate_n=41,
    )

    # concat data with mnist_subsets_all_165_190_3
    mnist_165 = Data.load("mnist_subsets_all_165_190_3")
    # concat dat with mnist_165, save as new file
    new = mx.concatenate([data, mnist_165], axis=0)
    print("SHAPE", new.shape)
    Data.save("all_subsets_MNIST_only_and_165_190_3", new)

    # Concatenate results for each n value
    for n in range(4, 8):
        base_data = Data.load(f"k_max_subsets_at_n{n}")
        base_data = base_data[:41]
        high_data = Data.load(f"k_max_mnist_subsets_all_165_190_3_n{n}")

        # Verify lengths
        assert (
            base_data.shape[0] == 41
        ), f"Base data for n={n} has length {base_data.shape[0]}, expected 41"
        assert (
            high_data.shape[0] == 12
        ), f"High data for n={n} has length {high_data.shape[0]}, expected 12"

        # Concatenate and save
        combined = mx.concatenate([base_data, high_data], axis=0)
        assert (
            combined.shape[0] == 53
        ), f"Combined data for n={n} has length {combined.shape[0]}, expected 53"

        Data.save(f"k_max_subsets_combined_n{n}", combined)

    plot_with_config(plot_inputs, "Scaling for MNIST")
