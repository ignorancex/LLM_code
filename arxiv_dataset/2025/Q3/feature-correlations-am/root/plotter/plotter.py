import contextlib
from dataclasses import dataclass
from functools import wraps

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

from matplotlib.cm import get_cmap

from root.old.experiments.MNIST.bottlenecks.verify.plot import fit_exp_line

pyplot_colors = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
]


@dataclass
class LinePlotInput:
    x: mx.array
    y: mx.array
    line_label: str


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


plt.rcParams.update(
    {
        "font.size": 18,  # Default font size for all text
        "axes.titlesize": 26,  # Title font size
        "axes.labelsize": 24,  # X and Y axis labels font size
        "xtick.labelsize": 20,  # X tick labels font size
        "ytick.labelsize": 20,  # Y tick labels font size
        "legend.fontsize": 16,  # Legend font size
    }
)


class Plotter:
    @staticmethod
    def plot_lines(
        datasets: list[LinePlotInput],
        title,
        x_label,
        y_label,
        y_scale="linear",
        fit_curve=False,
        save=False,
        xlim: tuple = None,
    ):
        # Get a color palette from a colormap (e.g., 'viridis')
        cmap = get_cmap("plasma")
        colors = [cmap(i / len(datasets)) for i in range(len(datasets))]
        for index in range(len(datasets)):
            # randomly choose a color from the list of colors
            dataset = datasets[index]
            color = colors[index]

            x = np.array(dataset.x)
            y = np.array(dataset.y)

            plt.scatter(
                dataset.x,
                dataset.y,
                color=color,
                alpha=0.7,
                s=10,
                label=dataset.line_label,
            )

            if fit_curve:
                # optional in case we want all curves to start at the same y-value (2)
                y_starting_point = 2
                y_max = 50

                # sort data for a nicer curve fit
                sorted_indexes = np.argsort(x)
                x = x[sorted_indexes]
                y = y[sorted_indexes]

                popt, _ = curve_fit(
                    lambda x, a, b, c: exp_func(x, a, b, c),
                    x,
                    y,
                    p0=(1, 0.01, 0),
                    maxfev=10000,
                )  # Initial guesses for a, b

                a, b, c = popt
                x_stop = (np.log(y_max - c) - np.log(a)) / b

                # Generate smooth points
                x_smooth = np.linspace(min(x), x_stop, 500)
                y_smooth = exp_func(x_smooth, a, b, c)
                # Plot the exponential curve
                plt.plot(x_smooth, y_smooth, color=color)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if xlim:
            plt.xlim(*xlim)

        if y_scale.startswith("log"):
            if ":" in y_scale:
                _, base = y_scale.split(":")
                y_scale = "log"
                if base == "e":
                    base = np.e
                else:
                    base = float(base)
                plt.yscale(y_scale, base=base)
            else:
                plt.yscale(y_scale)
        else:
            plt.yscale(y_scale)
        plt.legend(loc="upper left")
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig("plot.pdf", format="pdf", dpi=300)

        plt.show()

    @staticmethod
    def plot_lines_subplots(
        datasets_list: list[list[LinePlotInput]],
        titles: list[str],
        x_label,
        y_label,
        y_scale="linear",
        fit_curve=False,
        xlim: tuple = None,
        overall_title: str = None,
        save=False,
        legend_loc="upper left",
    ):
        """
        Plots multiple datasets in subplots.

        Args:
            datasets_list: List of lists of LinePlotInput objects for each subplot.
            titles: List of titles for each subplot.
            x_label: Common x-axis label.
            y_label: Common y-axis label.
            y_scale: Scale for y-axis.
            fit_curve: Whether to fit a curve to the data.
            xlim: Limits for x-axis.
            overall_title: Overall title for the entire figure.
        """
        num_subplots = len(datasets_list)
        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

        if overall_title:
            fig.suptitle(overall_title, fontsize=14)

        for i, (ax, datasets, title) in enumerate(zip(axes, datasets_list, titles)):
            cmap = get_cmap("plasma")
            colors = [cmap(i / len(datasets)) for i in range(len(datasets))]
            for index, dataset in enumerate(datasets):
                color = colors[index]
                x = np.array(dataset.x)
                y = np.array(dataset.y)

                ax.scatter(
                    x,
                    y,
                    color=color,
                    alpha=0.7,
                    s=10,
                    label=dataset.line_label,
                )

                if fit_curve and len(x) > 2:  # Changed from len(x) > 0 to len(x) > 2
                    sorted_indexes = np.argsort(x)
                    x = x[sorted_indexes]
                    y = y[sorted_indexes]

                    try:
                        popt, _ = curve_fit(
                            lambda x, a, b, c: exp_func(x, a, b, c),
                            x,
                            y,
                            p0=(1, 0.01, 0),
                            maxfev=10000,
                        )
                        a, b, c = popt
                        print(
                            f"Subplot {title}, {dataset.line_label}: a={a:.3f}, b={b:.3f}, c={c:.3f}"
                        )

                        # Calculate x_stop based on y_max = 50
                        y_max = 50
                        x_stop = (np.log(y_max - c) - np.log(a)) / b

                        # Generate smooth points
                        x_smooth = np.linspace(min(x), x_stop, 500)
                        y_smooth = exp_func(x_smooth, a, b, c)
                        ax.plot(x_smooth, y_smooth, color=color)
                    except (RuntimeError, ValueError) as e:
                        print(
                            f"Warning: Could not fit curve for {title}, {dataset.line_label}: {e}"
                        )
                elif fit_curve:
                    print(
                        f"Warning: Not enough points to fit curve for {title}, {dataset.line_label}. Need at least 3 points."
                    )

            ax.set_title(title)
            ax.set_xlabel(x_label)

            # Only show y-axis label for the first subplot
            if i == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")  # Remove label from other subplots
                ax.tick_params(labelleft=False)  # Remove y-axis tick labels

            if xlim:
                ax.set_xlim(*xlim)
            if y_scale.startswith("log"):
                ax.set_yscale(y_scale)
            ax.legend(loc=legend_loc)

        plt.tight_layout(
            rect=[0, 0.05, 1, 0.95]
        )  # Adjust layout to make space for suptitle

        if save:
            plt.savefig("plot.pdf", format="pdf", dpi=300)

        plt.show()

    @staticmethod
    def plot_lines_polyfit(
        datasets: list[LinePlotInput],
        title,
        x_label,
        y_label,
        y_scale="linear",
        fit_curve=False,
        degrees=[],
    ):
        for index in range(len(datasets)):
            # randomly choose a color from the list of colors
            color = pyplot_colors[index]
            dataset = datasets[index]

            x = np.array(dataset.x)
            y = np.array(dataset.y)

            if fit_curve:
                sorted_indexes = np.argsort(x)
                x = x[sorted_indexes]
                y = y[sorted_indexes]

                coefficients = np.polyfit(x, y, deg=degrees[index])
                p = np.poly1d(coefficients)

                x_smooth = np.linspace(min(x), max(x), 500)
                y_smooth = p(x_smooth)

                y_max = 50

                # clip
                y_smooth = np.clip(y_smooth, 0, y_max)

                # Plot the exponential curve
                plt.plot(x_smooth, y_smooth, label=dataset.line_label, color=color)

                plt.scatter(
                    dataset.x, dataset.y, label=dataset.line_label, color=color, s=10
                )
            else:
                plt.scatter(dataset.x, dataset.y, label=dataset.line_label, color=color)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if y_scale.startswith("log"):
            if ":" in y_scale:
                _, base = y_scale.split(":")
                y_scale = "log"
                if base == "e":
                    base = np.e
                else:
                    base = float(base)
                plt.yscale(y_scale, base=base)
            else:
                plt.yscale(y_scale)
        else:
            plt.yscale(y_scale)
        plt.legend()
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_img(img):
        plt.figure()
        w_mat = plt.imshow(img, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Img")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def record_stdout(func, file_path):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with open(file_path, "w") as o:
                with contextlib.redirect_stdout(o):
                    func(*args, **kwargs)

        return wrapped

    @staticmethod
    def plot_matrix_heatmap(data):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, cmap="coolwarm", center=0)
        plt.title("Matrix Heatmap")
        plt.show()
