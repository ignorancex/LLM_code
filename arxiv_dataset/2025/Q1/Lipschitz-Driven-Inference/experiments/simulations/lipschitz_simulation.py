import matplotlib.pyplot as plt
from pathlib import Path
from lipschitz_driven_inference.estimators import (
    NNLipschitzDrivenEstimator,
    Estimator,
)
import scipy.stats as stats
from typing import Dict, List
import numpy as np
from two_dim_shift import TwoDimensionalShiftExperiment
import matplotlib as mpl
from matplotlib.lines import Line2D
from simulation_utils import basic_parser


class TwoDimensionalShiftLipschitzExperiment(TwoDimensionalShiftExperiment):

    def run_single_simulation(
        self,
        seed: int,
        data_generation_kwargs: Dict,
        estimators: List[Estimator],
        estimator_kwargs: List[Dict],
    ) -> Dict:

        if seed % 50 == 0:
            print(f"Running seed {seed}")
        training_data, test_data = self.generate_data(
            seed=seed, **data_generation_kwargs
        )
        results = dict()
        # Compute "true" parameter
        true_param = self.get_true_parameter(test_data)
        # Build the estimators
        for estimator, estimator_kwarg in zip(estimators, estimator_kwargs):
            estimator_kwarg_copy = estimator_kwarg.copy()
            estimator_instance = estimator(
                training_data, test_data, **estimator_kwarg_copy
            )
            point_estimate = estimator_instance.point_estimate(self.dim)
            ci = estimator_instance.confidence_interval(self.dim, self.alpha)
            bias_bdd = estimator_instance.bias_bdd(self.dim)
            std_bdd = estimator_instance.randomness_std_bdd(self.dim) * stats.norm.ppf(
                1 - self.alpha / 2
            )
            results[f"L={estimator_kwarg_copy['lipschitz_bound']}"] = {
                "point_estimate": point_estimate,
                "ci": ci,
                "ci_contains_param": ci.lower <= true_param <= ci.upper,
                "bias_bdd": bias_bdd,
                "std_bdd": std_bdd,
            }
        return results

    def _plot_results(self):
        """
        Create a subplot consisting of:
        - CI Width vs Shift
        - Bias-Variance Tradeoff at Shift = 0.0
        - Bias-Variance Tradeoff at Shift = 0.8
        All plots are for our method with varying Lipschitz constants.
        The legend representing Lipschitz constants is placed to the left of the plots in a specified order.
        """

        # ----------------------- Setup and Data Aggregation -----------------------

        fontsize = 20  # Adjusted for better readability

        # Use latex for text rendering
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        # Extract Lipschitz bounds and corresponding methods
        lipschitz_bounds = [
            estimator_kwargs["lipschitz_bound"]
            for estimator_kwargs in self.estimator_kwargs
        ]
        methods = list(
            self.results[0][0].keys()
        )  # Assuming methods correspond to Lipschitz bounds

        # Pair each method with its Lipschitz bound
        methods_with_lb = list(zip(methods, lipschitz_bounds))

        # Sort methods by increasing Lipschitz bound for consistent plotting
        methods_with_lb_sorted = sorted(methods_with_lb, key=lambda x: x[1])
        methods_sorted, lipschitz_bounds_sorted = zip(*methods_with_lb_sorted)

        # Initialize dictionaries to store metrics
        method_ci_widths = {method: [] for method in methods_sorted}
        method_bias_bdds = {method: [] for method in methods_sorted}
        method_std_bdds = {method: [] for method in methods_sorted}
        method_coverages = {method: [] for method in methods_sorted}

        # Aggregate metrics across shifts and methods
        for shift, results in zip(self.all_shifts, self.results):
            for method in methods_sorted:
                # CI Width
                ci_width = np.mean(
                    [
                        result[method]["ci"][1] - result[method]["ci"][0]
                        for result in results
                    ]
                )
                method_ci_widths[method].append(ci_width)

                # Bias and Standard Deviation Bounds
                bias_bdd = np.mean([result[method]["bias_bdd"] for result in results])
                method_bias_bdds[method].append(bias_bdd)

                std_bdd = np.mean([result[method]["std_bdd"] for result in results])
                method_std_bdds[method].append(std_bdd)

                # Coverage
                coverage = np.mean(
                    [result[method]["ci_contains_param"] for result in results]
                )
                method_coverages[method].append(coverage)

        # ----------------------- Color Palette Setup -----------------------

        # Define a color-blind-friendly color palette
        colorblind_palette = mpl.colormaps.get_cmap("Paired").colors
        color_dict = {
            method: color for method, color in zip(methods_sorted, colorblind_palette)
        }

        # ----------------------- Figure and Subplots Setup -----------------------

        # Create the figure and subplots
        fig, axs = plt.subplots(
            1, 3, figsize=(20, 6), constrained_layout=True, sharey=True
        )
        plt.rcParams.update({"font.size": fontsize})

        # ----------------------- Subplot 1: CI Width vs Shift -----------------------

        ax1 = axs[0]
        ax1.set_title("CI Width vs Shift", fontsize=fontsize)
        ax1.set_xlim(np.min(self.all_shifts), np.max(self.all_shifts))
        ax1.set_xlabel("Shift", fontsize=fontsize)
        ax1.set_ylabel("CI Width", fontsize=fontsize)
        ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
        # set tick labels fontsize
        ax1.tick_params(axis="both", which="major", labelsize=fontsize - 4)

        # Plot CI Widths for each Lipschitz constant
        for method in methods_sorted:
            ax1.plot(
                self.all_shifts,
                method_ci_widths[method],
                label=f"Lipschitz: {method}",
                linestyle="-",
                linewidth=2,
                color=color_dict[method],
            )
            ax1.scatter(
                self.all_shifts,
                method_ci_widths[method],
                marker="o",
                s=50,
                color=color_dict[method],
                zorder=5,
            )

        # ----------------------- Subplot 2: Bias-Variance Tradeoff at Shift = 0.0 -----------------------

        ax2 = axs[1]
        ax2.set_title("Bias-Randomness Tradeoff (Shift = 0.0)", fontsize=fontsize)
        ax2.set_xlabel("Lipschitz Constant", fontsize=fontsize)
        ax2.grid(True, which="both", linestyle=":", linewidth=0.5)
        # set tick labels fontsize
        ax2.tick_params(axis="both", which="major", labelsize=fontsize - 4)

        # Extract metrics at Shift = 0.0 (assuming it's the first shift)
        # Adjust the index if 'shift=0.0' is not at position 0
        shift_index_0 = 4  # Change if necessary
        bias_bdds_shift_0 = [
            2 * method_bias_bdds[method][shift_index_0] for method in methods_sorted
        ]
        ci_widths_shift_0 = [
            method_ci_widths[method][shift_index_0] for method in methods_sorted
        ]
        std_bdds_shift_0 = [
            2 * method_std_bdds[method][shift_index_0] for method in methods_sorted
        ]

        # Plot Bias, CI, and Variance
        ax2.plot(
            lipschitz_bounds_sorted,
            ci_widths_shift_0,
            label="CI Width",
            marker="o",
            linestyle="-",
            linewidth=1,
            color="black",
        )
        ax2.plot(
            lipschitz_bounds_sorted,
            bias_bdds_shift_0,
            label="Bias",
            marker="x",
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax2.plot(
            lipschitz_bounds_sorted,
            std_bdds_shift_0,
            label="Randomness",
            marker="s",
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax2.fill_between(
            lipschitz_bounds_sorted,
            np.zeros(len(lipschitz_bounds_sorted)),
            ci_widths_shift_0,
            alpha=0.1,
            color="tab:orange",
        )

        # add a legend just for ax2
        ax2.legend(
            loc="upper left",
            frameon=False,
            handlelength=3,
            labelspacing=0.3,
            fontsize=0.9 * fontsize,
        )

        # ----------------------- Subplot 3: Bias-Variance Tradeoff at Shift = 0.8 -----------------------

        ax3 = axs[2]
        ax3.set_title("Bias-Randomness Tradeoff (Shift = 0.8)", fontsize=fontsize)
        ax3.set_xlabel("Lipschitz Constant", fontsize=fontsize)
        ax3.grid(True, which="both", linestyle=":", linewidth=0.5)
        # set tick labels fontsize
        ax3.tick_params(axis="both", which="major", labelsize=fontsize - 4)

        # Extract metrics at Shift = 0.8 (assuming it's the last shift)
        # Adjust the index if 'shift=0.8' is not at the last position
        shift_index_8 = -1  # Change if necessary
        bias_bdds_shift_8 = [
            method_bias_bdds[method][shift_index_8] for method in methods_sorted
        ]
        ci_widths_shift_8 = [
            method_ci_widths[method][shift_index_8] for method in methods_sorted
        ]
        std_bdds_shift_8 = [
            method_std_bdds[method][shift_index_8] for method in methods_sorted
        ]

        # Plot Bias, CI, and Tail Bound
        ax3.plot(
            lipschitz_bounds_sorted,
            ci_widths_shift_8,
            label="CI Width",
            marker="o",
            linestyle="-",
            linewidth=1,
            color="black",
        )
        ax3.plot(
            lipschitz_bounds_sorted,
            [2 * bias for bias in bias_bdds_shift_8],
            label="Bias",
            marker="x",
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax3.plot(
            lipschitz_bounds_sorted,
            [2 * std for std in std_bdds_shift_8],
            label="Randomness",
            marker="s",
            linestyle="--",
            linewidth=1,
            color="black",
        )
        ax3.fill_between(
            lipschitz_bounds_sorted,
            np.zeros(len(lipschitz_bounds_sorted)),
            ci_widths_shift_8,
            alpha=0.1,
            color="tab:orange",
        )

        # add a legend just for ax3
        ax3.legend(
            loc="upper left",
            frameon=False,
            handlelength=3,
            labelspacing=0.3,
            fontsize=0.9 * fontsize,
        )

        # ----------------------- Legend Placement -----------------------

        # Define the desired legend order
        desired_legend_order = [10, 7.5, 5, 0.1, 3.5, 0.5, 1, 2]

        # Create custom legend handles based on desired order
        legend_elements = []
        for lb in desired_legend_order:
            # Find the method corresponding to the Lipschitz bound
            try:
                idx = lipschitz_bounds_sorted.index(lb)
                method = methods_sorted[idx]
                legend_elements.append(
                    Line2D([0], [0], color=color_dict[method], lw=2, label=lb)
                )
            except ValueError:
                # If the Lipschitz bound is not present, skip it
                continue

        # Position the legend to the left of the subplots with minimal white space
        fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            frameon=False,
            title="Lipschitz Constants",
            handlelength=3,
            labelspacing=0.3,
            fontsize=fontsize,
        )

        # ----------------------- Final Layout Adjustments -----------------------

        # Adjust layout to make room for the legend on the left
        plt.tight_layout(rect=[0.12, 0.12, 0.9, 0.9])

        # ----------------------- Save the Figure -----------------------

        # Save the figure with tight bounding to include the legend

        plt.savefig(Path(self.results_dir, "combined_plot.pdf"), bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    # Parse arguments using argparse
    parser = basic_parser()
    parser.add_argument(
        "--lipschitz_bounds",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10],
    )
    args = parser.parse_args()
    # Set up the experiment
    data_generation_kwargs = {
        "n": args.n,
        "m": args.m,
        "p": 2,
        "noise_std": args.noise_std,
        "include_intercept": True,
    }
    all_shifts = [
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0,
        0.2,
        0.4,
        0.6,
        0.8,
    ]  # range of shift values. Should be in [0, 1]
    file_path = Path(__file__).parent
    results_dir = f"results/lipschitz_simulation/n={args.n}_p={1}_m={args.m}_noise_std={args.noise_std}"
    results_dir = str(Path(file_path, results_dir))
    experiment = TwoDimensionalShiftLipschitzExperiment(
        name="lipschitz_simulation",
        results_dir=results_dir,
        dim=1,  # dimension of interest. This is X_1 since there is an intercept
        num_seeds=args.num_seeds,
        alpha=0.05,
        parallel_threads=args.num_threads,
        data_generation_kwargs=data_generation_kwargs,
        estimators=[NNLipschitzDrivenEstimator] * len(args.lipschitz_bounds),
        estimator_kwargs=[
            {
                "num_neighbors": args.num_neighbors,
                "lipschitz_bound": lipbdd,
            }
            for lipbdd in args.lipschitz_bounds
        ],
        is_gls=False,
        all_shifts=all_shifts,
    )
    # Run the experiment
    experiment.run()
    # Save the results
    experiment.save_results()
    # Plot the results
    experiment.plot_results()
