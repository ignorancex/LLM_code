import matplotlib.pyplot as plt
import numpy as np

from root.correlations.correlations import Correlations
from root.data.data import Data
from root.plotter.plotter import LinePlotInput, Plotter


def create_x(y, datasets):
    x = []
    for i in range(datasets.shape[0]):
        idx = int(y[i])
        x.append(Correlations.calc_average_hd(datasets[i, :idx, :]))
    return x


# remove all values above 50 if they are consecutive
# do this for every x and y combination
def remove_consecutive_values(x, y, threshold=50, once=False):
    new_x, new_y = [], []
    adj_y = []
    for val in y:
        new_v = float(val)
        adj_y.append(new_v)
    y = adj_y
    above_threshold = False
    for xi, yi in zip(list(x), list(y)):
        if yi >= threshold:
            if once:
                break
            if not above_threshold:
                new_x.append(xi)
                new_y.append(yi)
            above_threshold = True
        else:
            new_x.append(xi)
            new_y.append(yi)
            above_threshold = False
    return new_x, new_y


def truncate_data(x, y, n=10):
    """Truncate x and y arrays to first n values"""
    return x[:n], y[:n]


def prepare_plot_data(data_file, k_max_files, threshold=50, truncate_n=None):
    """
    Prepares data for plotting by loading files, creating x values,
    removing consecutive values above threshold, and optionally truncating.

    Args:
        data_file: str, name of dataset file to load
        k_max_files: list of (str, str) tuples containing (k_max_file, label)
        threshold: int, threshold for removing consecutive values
        truncate_n: int or None, number of points to truncate to

    Returns:
        list of LinePlotInput objects ready for plotting
    """
    datasets = Data.load(data_file)
    plot_inputs = []

    for k_max_file, label in k_max_files:
        # Load y values and create corresponding x values
        y = Data.load(k_max_file)
        x = create_x(y, datasets)

        # Remove consecutive values above threshold
        x, y = remove_consecutive_values(x, y, threshold=threshold)

        # Truncate if specified
        if truncate_n is not None:
            x, y = truncate_data(x, y, n=truncate_n)

        plot_inputs.append(LinePlotInput(x=x, y=y, line_label=label))

    return plot_inputs


def plot_with_config(
    plot_inputs,
    title,
    x_label="Mean Separation (bits)",
    y_label="Memory Capacity (K_max)",
    y_scale="linear",
    fit_curve=True,
    save=False,
    xlim=None,
):
    """
    Creates plot with given inputs and configuration
    """
    Plotter.plot_lines(
        plot_inputs,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_scale=y_scale,
        fit_curve=fit_curve,
        save=save,
        xlim=xlim,
    )


###
# Helper function for Figure 3
###
def gather_kmax_in_bucket(
    dataset_name, result_name, comparison_hd, n, hd_threshold=10.0
):
    """
    1. Loads the dataset (array of subsets),
    2. Finds *all* subsets whose mean HD is within +/- hd_threshold of comparison_hd,
    3. Loads the corresponding K_max array for polynomial degree n,
    4. Averages the K_max values for those subsets.

    Returns that average. If no subsets found, returns 0 or some placeholder.
    """

    # 1) Load the subsets and their K_max values
    subsets = Data.load(dataset_name)  # shape: (num_subsets, ...)
    k_maxes = Data.load(f"{result_name}{n}")  # shape: (num_subsets, )

    # Replace 0s with 50s if they follow two consecutive 50s
    # for i in range(2, len(k_maxes)):
    #     if k_maxes[i - 2] == 50 and k_maxes[i - 1] == 50 and k_maxes[i] == 0:
    #         k_maxes[i] = 50

    # Make sure subsets.shape[0] == k_maxes.shape[0]
    # so index i in subsets corresponds to index i in k_maxes

    # fill up the k_maxes with 50s if the last value is a 50 and the array is not
    # up until the same length as subsets.shape[0]
    # max_length = subsets.shape[0]
    # if k_maxes[-1] == 50 and len(k_maxes) < max_length:
    #     k_maxes = mx.concat(
    #         [k_maxes, mx.array([50 for _ in range(len(k_maxes), max_length)])]
    #     )

    # 2) Collect all indices that fall in the desired HD bucket
    indices_in_bucket = []
    for i in range(subsets.shape[0]):
        # Only calculate HD up to k_max for this subset
        k_max = int(k_maxes[i])
        if k_max == 50:  # Skip if k_max is the sentinel value
            continue

        # Calculate mean HD using only the first k_max patterns
        subset_up_to_kmax = subsets[i, :k_max, :]
        mean_hd = Correlations.calc_average_hd(subset_up_to_kmax)

        if abs(mean_hd - comparison_hd) <= hd_threshold:
            indices_in_bucket.append(i)

    # 3) Gather their K_max
    bucket_kmax_values = []
    for idx in indices_in_bucket:
        if k_maxes[idx] < 50:  # skip any sentinel 50
            bucket_kmax_values.append(k_maxes[idx])

    # 4) Return the average or median
    if len(bucket_kmax_values) == 0:
        return 0  # or np.nan, or some fallback
    else:
        return float(np.mean(bucket_kmax_values))


####
# FIGURE 1A
####
def plot_new_mnist():
    plot_inputs = prepare_plot_data(
        "all_subsets_MNIST_only_and_165_190_3",
        [
            ("k_max_subsets_combined_n6", "n=6"),
            ("k_max_subsets_combined_n7", "n=7"),
            ("k_max_subsets_combined_n8", "n=8"),
        ],
        threshold=50,
    )
    plot_with_config(plot_inputs, "MNIST", save=True, xlim=(0, 400))


####
# FIGURE 1B
####
def plot_artificial():
    plot_inputs = prepare_plot_data(
        "decreasingly_correlated_sets",
        [
            # ("k_max_decor_n2", "n=2"),
            # ("k_max_decor_n3", "n=3"),
            # ("k_max_decor_n4", "n=4"),
            ("k_max_decor_n6", "n=6"),
            ("k_max_decor_n7", "n=7"),
            ("k_max_decor_n8", "n=8"),
        ],
        threshold=50,
    )
    plot_with_config(
        plot_inputs, "Biased Rademacher Patterns", save=True, xlim=(0, 400)
    )


####
# FIGURE 2
####
def plot_mnist_vs_artificial_correlations():
    # n=7 comparison
    x1 = prepare_plot_data(
        "all_subsets_MNIST_only_and_165_190_3",
        [("k_max_subsets_combined_n7", "mnist")],
    )
    x2 = prepare_plot_data(
        "decreasingly_correlated_sets",
        [("k_max_decor_n7", "artificial")],
    )

    # n=11 comparison
    x3 = prepare_plot_data(
        "all_subsets_MNIST_only_and_165_190_3",
        [("k_max_subsets_combined_n9", "mnist")],
    )
    x4 = prepare_plot_data(
        "decreasingly_correlated_sets",
        [("k_max_decor_n9", "artificial")],
    )

    # n=16 comparison
    x5 = prepare_plot_data(
        "all_subsets_MNIST_only_and_165_190_3",
        [("k_max_subsets_combined_n11", "mnist")],
    )
    x6 = prepare_plot_data(
        "decreasingly_correlated_sets",
        [("k_max_decor_n11", "artificial")],
    )

    Plotter.plot_lines_subplots(
        datasets_list=[
            x2 + x1,  # Combine MNIST and artificial for n=7
            x4 + x3,  # Combine MNIST and artificial for n=11
            x6 + x5,  # Combine MNIST and artificial for n=16
        ],
        titles=["n=7", "n=9", "n=11"],
        x_label="Mean Separation (bits)",
        y_label="Memory Capacity (K_max)",
        xlim=(0, 400),
        fit_curve=True,
        save=True,
        legend_loc="lower right",
    )


####
# FIGURE 3
####
def plot_n_to_kmax():
    """
    We create a plot of n vs. average memory capacity (K_max) for
    different dataset buckets around a specified mean Hamming distance (HD).
    Instead of using exactly one subset, we aggregate all subsets that
    have HD within +/- some threshold of 'comparison_hd' and average
    their K_max scores.

    We'll do this for both an 'artificial' dataset and an 'mnist' dataset.
    """
    artificial = "decreasingly_correlated_sets"
    mnist = "all_subsets_MNIST_only_and_165_190_3"

    # You can list multiple HD targets if desired
    target_hds = [60, 90, 120]

    # Polynomial degrees of interest
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, *list(range(14, 39, 2))]

    plot_data = []
    for hd in target_hds:
        # We'll build Y-values for each dataset separately
        y_artificial = []
        y_mnist = []

        for n in degrees:
            # For each n, gather all subsets near hd and compute an average K_max
            avg_kmax_art = gather_kmax_in_bucket(
                dataset_name=artificial,
                result_name="k_max_decor_n",
                comparison_hd=hd,
                n=n,
                hd_threshold=10.0,  # You can tweak this
            )
            avg_kmax_mnist = gather_kmax_in_bucket(
                dataset_name=mnist,
                result_name="k_max_subsets_combined_n",
                comparison_hd=hd,
                n=n,
                hd_threshold=10.0,
            )

            # Fill in 50 for zero values
            if avg_kmax_art == 0:
                avg_kmax_art = 50
            if avg_kmax_mnist == 0:
                # 25 if 50 and 0, but following zeros should be 50 anyway
                avg_kmax_mnist = 50

            y_artificial.append(avg_kmax_art)
            y_mnist.append(avg_kmax_mnist)

            # Debugging output
            # print(
            #     f"HD={hd}, n={n}, Artificial K_max={avg_kmax_art}, MNIST K_max={avg_kmax_mnist}"
            # )

        x1 = degrees.copy()
        x2 = degrees.copy()

        # x1, y1 = remove_consecutive_values(x1, y_artificial, threshold=49, once=True)
        # x2, y2 = remove_consecutive_values(x2, y_mnist, threshold=49, once=True)

        # print(x1, y1)
        # print(x2, y2)

        # Only add datasets if they have data points
        # if len(x1) > 0 and len(y1) > 0:
        #     plot_data.append(LinePlotInput(x=x1, y=y1, line_label=f"Artificial"))
        # if len(x2) > 0 and len(y2) > 0:
        #     plot_data.append(LinePlotInput(x=x2, y=y2, line_label=f"MNIST"))

        x_no50 = []
        y_no50 = []
        for xi, yi in zip(x1, y_artificial):
            if yi < 50:  # or <= 49 if you prefer
                x_no50.append(xi)
                y_no50.append(yi)

        x_no50_mnist = []
        y_no50_mnist = []
        for xi, yi in zip(x2, y_mnist):
            if yi < 50:  # or <= 49 if you prefer
                x_no50_mnist.append(xi)
                y_no50_mnist.append(yi)

        # Debugging output after removing consecutive values
        print(
            f"After removal - HD={hd}, Artificial: {list(zip(x_no50, y_no50))}, MNIST: {list(zip(x_no50_mnist, y_no50_mnist))}"
        )

        # Create lines for plotting
        plot_data.append(LinePlotInput(x=x_no50, y=y_no50, line_label=f"Artificial"))
        plot_data.append(
            LinePlotInput(x=x_no50_mnist, y=y_no50_mnist, line_label=f"MNIST")
        )

    # Now we have lines that reflect an *average* capacity at each (n, HD) pair
    Plotter.plot_lines_subplots(
        datasets_list=[plot_data[i * 2 : (i + 1) * 2] for i in range(len(target_hds))],
        titles=[f"HD≈{hd}" for hd in target_hds],
        x_label="Polynomial Degree (n)",
        y_label="Memory Capacity (K_max)",
        xlim=(min(degrees), max(degrees)),
        fit_curve=True,  # or False, up to you
        save=True,  # or True, if you want to save the plot
        legend_loc="lower right",
    )


def main():
    # # FIGURE 1A
    # plot_artificial()

    # # FIGURE 1B
    # plot_new_mnist()

    # # FIGURE 2
    plot_mnist_vs_artificial_correlations()

    # FIGURE 3
    # plot_n_to_kmax()


if __name__ == "__main__":
    main()
