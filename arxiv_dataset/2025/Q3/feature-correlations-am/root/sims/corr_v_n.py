from root.correlations.correlations import Correlations
from root.data.data import Data
from root.plotter.plotter import LinePlotInput, Plotter


def plot_data():
    data = Data.load("all_subsets")
    y = Data.load("memcaps")

    x1 = []
    x2 = []
    x3 = []
    for i in range(data.shape[0]):
        x1.append(Correlations.calc_average_hd(data[i]))
        x2.append(Correlations.calc_min_hd(data[i]))
        x3.append(Correlations.calc_average_dcor(data[i], inverse=True))

    Plotter.plot_lines_with_multiple_aligned_x_axes(
        [
            LinePlotInput(x=x1, y=y, line_label="Average HD"),
            LinePlotInput(x=x2, y=y, line_label="Minimum HD"),
            LinePlotInput(x=x3, y=y, line_label="Average Dcor"),
        ],
        title="Data-dependent interaction function scaling",
        x_label="Hamming Distance",
        y_label="n at 100%",
    )


def main():
    plot_data()


if __name__ == "__main__":
    main()
