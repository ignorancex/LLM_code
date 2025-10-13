import matplotlib.pyplot as plt
import numpy as np

min_HD = np.array([39, 6, 56, 36, 36, 36, 37, 23, 30, 38])
mean_HD = np.array([123, 62, 131, 116, 104, 118, 118, 101, 122, 93])

y_axis = [15, 53, 13, 17, 20, 13, 18, 21, 16, 19]


def make_sorted_set(hds, memcaps):
    dataset = np.vstack((hds, memcaps))
    dataset = dataset[:, dataset[0, :].argsort()]
    return dataset


def fit_exp_line(x, y):
    fitted_line = np.polyfit(x, y, 1)
    new_y = fitted_line[1] + fitted_line[0] * x
    return new_y


def plot_scatter_with_fitted_line(dataset, label, xlabel, ylabel, color, no_xlim=False):
    plt.scatter(dataset[0, :], dataset[1, :], label=label, c=color)
    # ynew = fit_exp_line(dataset[0, :], dataset[1, :])
    # plt.plot(dataset[0, :], ynew, label="Fitted line", c=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim((0, 60))
    if not no_xlim:
        plt.xlim((0, 390))
    plt.legend(loc="upper right")
    plt.show()


def main():

    min_set = make_sorted_set(min_HD, y_axis)
    mean_set = make_sorted_set(mean_HD, y_axis)

    print(min_set)
    print(mean_set)

    plot_scatter_with_fitted_line(
        min_set,
        "MIN Hamming Distance vs. Memorization Capacity",
        "Min HD",
        "Mem Cap",
        "red",
    )
    # plot_scatter_with_fitted_line(mean_set, "MEAN Hamming Distance vs. Memorization Capacity", "Min HD", "Mem Cap", "blue")


if __name__ == "__main__":
    main()
