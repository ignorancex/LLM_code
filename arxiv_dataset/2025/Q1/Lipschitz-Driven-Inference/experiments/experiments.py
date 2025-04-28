from pathlib import Path
import os
from abc import abstractmethod
from typing import List, Dict
import json_tricks
import numpy as np
import matplotlib.pyplot as plt
from lipschitz_driven_inference.estimators import Estimator
from joblib import Parallel, delayed
from statsmodels.regression.linear_model import OLS

style_dict = {
    'Ours': {'linestyle': '-', 'linewidth': 3, 'color': 'black', 'marker': 'o'},
    'GP BCIs': {'linestyle': '--', 'linewidth': 1.5, 'color': 'C0', 'marker': 'x'},
    'KDEIW': {'linestyle': '-.', 'linewidth': 1.5, 'color': 'C1', 'marker': 'x'},
    'Sandwich': {'linestyle': ':', 'linewidth': 1.5, 'color': 'C2', 'marker': 'x'},
    'OLS': {'linestyle': (0, (5, 10)), 'linewidth': 1.5, 'color': 'C3', 'marker': 'x'},
    'GLS': {'linestyle': (0, (3, 5, 1, 5)), 'linewidth': 1.5, 'color': 'C4', 'marker': 'x'}
}

class Experiment:
    def __init__(self, name: str, results_dir: Path, dim: int, alpha: float, **kwargs):
        self.name = name
        self.results_dir = results_dir
        self.results = None
        self.dim = dim
        self.alpha = alpha

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def load_results(self) -> None:
        """
        Load the results from the results directory.
        """
        try:
            results = json_tricks.load(str(Path(self.results_dir, "results.json")))
        except FileNotFoundError:
            results = None
            print(
                f"Results not found for {self.name} in {self.results_dir}. Try running the experiment first."
            )
        self.results = results

    def saved_results_exist(self) -> bool:
        return Path(self.results_dir, "results.json").exists()

    def save_results(self) -> None:
        if self.results is None:
            print("No results to save.")
            return
        os.makedirs(self.results_dir, exist_ok=True)
        json_tricks.dump(self.results, str(Path(self.results_dir, "results.json")))

    def plot_results(self) -> None:
        if self.results is None:
            print("No results to plot.")
        else:
            self._plot_results()

    @abstractmethod
    def _plot_results(self):
        raise NotImplementedError


class SimulationExperiment(Experiment):

    def __init__(
        self,
        name,
        results_dir: Path,
        dim: int,
        alpha: float,
        num_seeds: int,
        data_generation_kwargs: Dict,
        estimators: List[Estimator],
        estimator_kwargs: List[Dict],
        parallel_threads: int = 5,
        all_shifts=[-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
        **kwargs,
    ):
        super().__init__(name, results_dir, dim, alpha, **kwargs)
        self.num_seeds = num_seeds
        self.data_generation_kwargs = data_generation_kwargs
        self.estimators = estimators
        self.estimator_kwargs = estimator_kwargs
        self.parallel_threads = parallel_threads
        self.all_shifts = all_shifts

    def get_true_parameter(self, test_data):
        return OLS(test_data.y[:, None], test_data.X).fit().params[self.dim]

    @abstractmethod
    def generate_data(self, seed: int, **kwargs):
        raise NotImplementedError

    def run_simulation(
        self,
        data_generation_kwargs,
    ) -> List[Dict]:
        return Parallel(n_jobs=self.parallel_threads)(
            delayed(self.run_single_simulation)(
                seed,
                data_generation_kwargs,
                self.estimators,
                self.estimator_kwargs,
            )
            for seed in range(self.num_seeds)
        )

    def run_single_simulation(
        self,
        seed: int,
        data_generation_kwargs: Dict,
        estimators: List[Estimator],
        estimator_kwargs: List[Dict],
    ) -> Dict:
        training_data, test_data = self.generate_data(
            seed=seed, **data_generation_kwargs
        )
        results = dict()
        # Compute "true" parameter
        true_param = self.get_true_parameter(test_data)
        # Build the estimators
        for estimator, estimator_kwarg in zip(estimators, estimator_kwargs):
            estimator_kwarg_copy = estimator_kwarg.copy()
            if "kernel" in estimator_kwarg.keys():
                kernel_name = estimator_kwarg_copy.pop("kernel")
                kernel_kwargs = estimator_kwarg_copy.pop("kernel_kwargs")
                kernel_instance = kernel_name(
                    **kernel_kwargs, active_dims=np.arange(training_data.S.shape[1])
                )
                estimator_kwarg_copy.update({"kernel": kernel_instance})

            estimator_instance = estimator(
                training_data, test_data, **estimator_kwarg_copy
            )
            point_estimate = estimator_instance.point_estimate(self.dim)
            ci = estimator_instance.confidence_interval(self.dim, self.alpha)
            results[estimator_instance.name()] = {
                "point_estimate": point_estimate,
                "ci": ci,
                "ci_contains_param": ci.lower <= true_param <= ci.upper,
            }
        return results

    def run(self):
        all_results = list()
        for i, shift in enumerate(self.all_shifts):
            print("Running shift: ", shift)
            print("Progress", i / len(self.all_shifts), "%")
            data_generation_kwargs = self.data_generation_kwargs.copy()
            data_generation_kwargs.update({"shift": shift})  # update the shift
            results = self.run_simulation(
                data_generation_kwargs,
            )
            all_results.append(results)
        self.results = all_results

    def _plot_results(self):
        """
        Make a subplot of the coverages and CI widths, share the legend as a separate axis on the right
        """

        fontsize = 20 

        # Define the desired legend order
        desired_order = ["Ours", "GP BCIs", "KDEIW", "Sandwich", "OLS", "GLS"]
        
        # Initialize methods and data structures
        methods = [estimator.name() for estimator in self.estimators]
        
        # Reorder methods based on desired_order
        methods_sorted = [method for method in desired_order if method in methods]
        
        method_ci_widths = {method: [] for method in methods_sorted}
        method_coverages = {method: [] for method in methods_sorted}

        # Calculate CI widths and coverages
        for shift, results in zip(self.all_shifts, self.results):
            for method in methods_sorted:
                # CI Width
                ci_width = np.mean(
                    [result[method]["ci"][1] - result[method]["ci"][0] for result in results]
                )
                method_ci_widths[method].append(ci_width)
                # Coverage
                coverage = np.mean(
                    [result[method]["ci_contains_param"] for result in results]
                )
                method_coverages[method].append(coverage)

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1]})
        plt.rcParams.update({"font.size": fontsize})

        # Plot Coverages
        axs[0].set_xlim(np.min(self.all_shifts), np.max(self.all_shifts))
        for method in methods_sorted:
            axs[0].plot(
                self.all_shifts,
                method_coverages[method],
                label=method,
                linestyle=style_dict[method]['linestyle'],
                linewidth=style_dict[method]['linewidth'],
                color=style_dict[method]['color'],
            )
            axs[0].scatter(
                self.all_shifts,
                method_coverages[method],
                marker=style_dict[method]['marker'],
                s=50,
                color=style_dict[method]['color']
            )
        # Add nominal coverage line
        axs[0].axhline(
            1 - self.alpha,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Nominal Coverage"
        )
        axs[0].set_ylim(-0.05, 1.05)
        axs[0].set_xlabel("Shift", fontsize=fontsize)
        axs[0].set_ylabel("Coverage", fontsize=fontsize)
        axs[0].grid(True, which='both', linestyle=':', linewidth=0.5)
        axs[0].set_title("Coverages", fontsize=fontsize)

        # Plot CI Widths
        axs[1].set_xlim(np.min(self.all_shifts), np.max(self.all_shifts))
        max_ci_width = np.max([np.max(widths) for widths in method_ci_widths.values()])
        for method in methods_sorted:
            axs[1].plot(
                self.all_shifts,
                method_ci_widths[method],
                label=method,
                linestyle=style_dict[method]['linestyle'],
                linewidth=style_dict[method]['linewidth'],
                color=style_dict[method]['color']
            )
            axs[1].scatter(
                self.all_shifts,
                method_ci_widths[method],
                marker=style_dict[method]['marker'],
                s=50,
                color=style_dict[method]['color']
            )
        axs[1].set_ylim(0, max_ci_width + 0.05)
        axs[1].set_xlabel("Shift", fontsize=fontsize)
        axs[1].set_ylabel("CI Width", fontsize=fontsize)
        axs[1].grid(True, which='both', linestyle=':', linewidth=0.5)
        axs[1].set_title("CI Widths", fontsize=fontsize)

        # Create a unified legend
        # Collect handles and labels from the first subplot
        handles, labels = axs[0].get_legend_handles_labels()
        
        # To ensure "Nominal Coverage" is after ours and GP BCIs
        nominal_handle = None
        nominal_label = "Nominal Coverage"
        if nominal_label in labels:
            idx = labels.index(nominal_label)
            nominal_handle = handles.pop(idx)
            labels.pop(idx)
        
        # Reorder handles and labels based on desired_order
        ordered_handles = []
        ordered_labels = []
        for method in methods_sorted:
            if method in labels:
                idx = labels.index(method)
                ordered_handles.append(handles.pop(idx))
                ordered_labels.append(labels.pop(idx))
        
        # Find place where to add nominal coverage by checking whether coverage at shift 0 of the methods are larger than nominal coverage
        nominal_coverage = 1 - self.alpha
        for i, method in enumerate(methods_sorted):
            if method_coverages[method][0] < nominal_coverage:
                break
        
        # Add nominal coverage in the third position after ours and GP BCIs
        if nominal_handle is not None:
            ordered_handles.insert(i, nominal_handle)
            ordered_labels.insert(i, nominal_label)

        
        # Place the legend to the left of the subplots
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc='center left',
            bbox_to_anchor=(-0.05, 0.5),
            frameon=False,
            title="Methods",
            handlelength=2,
            labelspacing=0.8
        )

        # Adjust layout to make room for the legend on the left
        # plt.tight_layout(rect=[0.2, 0, 1, 1])  # Leave space on the left for the legend
        plt.tight_layout(rect=[0.16, 0, 1, 1])

        # Save the combined plot
        plt.savefig(Path(self.results_dir, "combined_plot.pdf"), bbox_inches='tight')
        plt.close()


class RealDataExperiment(Experiment):

    def __init__(
        self,
        name,
        results_dir: Path,
        alpha: float,
        estimators: List[Estimator],
        estimator_kwargs: List[Dict],
        num_seeds: int = 1,
        seeds_to_plot: int = 1,
        parallel_threads: int = 1,
    ):
        super().__init__(name, results_dir, None, alpha)
        self.estimators = estimators
        self.estimator_kwargs = estimator_kwargs
        self.num_seeds = num_seeds
        self.parallel_threads = parallel_threads
        self.seeds_to_plot = seeds_to_plot

    def run(self):
        """
        Instead of running just once, we do multiple runs,
        each with a different seed for the subsample.
        """
        self.results = self.run_simulation()  # list of results

    def run_simulation(self):
        """
        Run multiple seeds in parallel, storing a list of results
        (one for each seed). This mirrors SimulationExperiment's usage.
        """
        seeds = range(self.num_seeds)
        results_list = Parallel(n_jobs=self.parallel_threads)(
            delayed(self.run_single_simulation)(seed) for seed in seeds
        )
        return results_list

    def run_single_simulation(self, seed: int) -> Dict[str, Dict]:
        """
        1) Generate data for a given seed (random subsample).
        2) Fit estimators, get point estimates, intervals, etc.
        """
        # Pass the seed so generate_data can do the random subsampling
        training_data, test_data = self.generate_data(seed=seed)
        print(f"Seed {seed}")
        dims = training_data.X.shape[1]
        # Compute the "true" parameter
        true_param = self.get_true_parameter(test_data)
        true_ci = self.get_true_ci(test_data)
        true_sd = self.get_true_sd(test_data)
        # Build the estimators and collect results
        results = {}
        for estimator, estimator_kwarg in zip(self.estimators, self.estimator_kwargs):
            estimator_kwarg_copy = estimator_kwarg.copy()
            if "kernel" in estimator_kwarg.keys():
                kernel_name = estimator_kwarg_copy.pop("kernel")
                kernel_kwargs = estimator_kwarg_copy.pop("kernel_kwargs")
                kernel_instance = kernel_name(
                    **kernel_kwargs, active_dims=np.arange(training_data.S.shape[1])
                )
                estimator_kwarg_copy.update({"kernel": kernel_instance})

            # Save the ground truth in the results 
            for d in range(dims):
                results[("Ground Truth", d)] = {
                    "point_estimate": true_param[d],
                    "ci": true_ci[d],
                    "sd": true_sd[d],
                }

            estimator_instance = estimator(
                training_data, test_data, **estimator_kwarg_copy
            )
            for d in range(dims):
                point_estimate = estimator_instance.point_estimate(d)
                ci = estimator_instance.confidence_interval(d, self.alpha)
                sd = estimator_instance.sd_estimate(d)
                results[(estimator_instance.name(), d)] = {
                    "point_estimate": point_estimate,
                    "ci": ci,
                    "ci_contains_param": (ci.lower <= true_param[d] <= ci.upper),
                    "ci_overlaps_true_ci": self.check_intervals_overlaps(
                        ci.lower, ci.upper, true_ci[d][0], true_ci[d][1]
                    ),
                    "ci_diff_contains_zero": self.check_interval_diff_contains_zero(
                        d, estimator_instance, point_estimate, sd, true_param, true_sd
                    ),
                }

        return results

    def check_intervals_overlaps(self, ci1_lower, ci1_upper, ci2_lower, ci2_upper):
        return not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)

    def check_interval_diff_contains_zero(
        self, dim, estimator_instance, point_estimate, sd_estimate, true_param, true_sd
    ):
        difference_estimate = point_estimate - true_param[dim]
        sd_difference = np.sqrt(sd_estimate**2 + true_sd[dim] ** 2)
        ci_lower = difference_estimate - 1.96 * sd_difference
        ci_upper = difference_estimate + 1.96 * sd_difference
        diff_contains_zero = ci_lower <= 0 <= ci_upper
        if estimator_instance.name() == "Ours":
            bias = estimator_instance.bias_bdd(dim)
            ci_low = difference_estimate - bias - 1.96 * sd_difference
            ci_up = difference_estimate + bias + 1.96 * sd_difference
            diff_contains_zero = ci_low <= 0 <= ci_up
        return diff_contains_zero

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError

    def get_true_parameter(self, test_data):
        true_params = OLS(test_data.y[:, None], test_data.X).fit().params
        return true_params

    def get_true_ci(self, test_data):
        """
        Compute the true confidence intervals for each dimension.
        """
        true_ols = (
            OLS(
                test_data.y[:, None],
                test_data.X,
            )
            .fit(cov_type="hc1")
            .conf_int(alpha=self.alpha)
        )

        true_ci = []
        for p in range(test_data.X.shape[1]):
            true_ci.append((true_ols[p, 0], true_ols[p, 1]))
        return true_ci

    def get_true_sd(self, test_data):
        """
        Compute the true standard deviation for each dimension.
        """
        true_ols = (
            OLS(
                test_data.y[:, None],
                test_data.X,
            )
            .fit(cov_type="hc1")
            .bse
        )
        return true_ols

    def save_results(self) -> None:
        """
        self.results is now a list of length num_seeds*num_dims.
        Each element is a dictionary (like {(estimator_name,dim): {...}, ...}).
        """
        if self.results is None:
            print("No results to save.")
            return
        os.makedirs(self.results_dir, exist_ok=True)
        # convert the dictionary tuple keys into nested dicts
        results_dict = {}
        for current_seed, seed_results in enumerate(self.results):
            results_dict[current_seed] = {}
            for key, value in seed_results.items():
                if key[0] not in results_dict[current_seed]:
                    results_dict[current_seed][key[0]] = {key[1]: value}
                else:
                    results_dict[current_seed][key[0]][key[1]] = value

        # go through results dict and check if the values are in a list (they should not)
        for key, value in results_dict.items():
            for method, method_results in value.items():
                for dim, dim_results in method_results.items():
                    for k, v in dim_results.items():
                        # check if v is an iterable
                        if isinstance(v, list):
                            print("IN A LIST")
                            for i in range(len(v)):
                                print(v)

        json_tricks.dump(results_dict, str(Path(self.results_dir, "results.json")))

    def plot_coverage_results(self, conservative: bool = False):
        """
        Produces one figure with subplots of shape (2 x n_dims):
        - The top row is coverage bar charts (one column per dimension)
        - The bottom row is average length bar charts (one column per dimension)
        """
        if self.results is None or len(self.results) == 0:
            print("No results to plot.")
            return

        # Set font size and latex format
        fontsize = 20
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("font", size=fontsize)

        # (Optional) Import saved data if needed
        if isinstance(self.results, str) or not isinstance(self.results, dict):
            self.results = json_tricks.load(str(Path(self.results_dir, "results.json")))

        # -------------------------------------------------------------
        # 1) Gather dimension keys & methods
        # -------------------------------------------------------------
        all_dims_in_json = list(
            self.results["0"]["Ours"].keys()
        )  # e.g. ['0','1','2','3']
        all_dims_in_json.sort(key=int)  # numeric sort
        dims_list = [d for d in all_dims_in_json if d != "0"]  # skip the intercept

        first_seed_result = self.results[
            "0"
        ]  # => {'MethodA': {'0': {...}, '1': {...}}, ...}
        all_methods = list(first_seed_result.keys())
        # Drop ground truth 
        all_methods = [method for method in all_methods if method != "Ground Truth"]
        all_methods.sort()

        # Define a lookup dict to rename dimensions
        dim_title_map = {1: "Aridity Index", 2: "Elevation", 3: "Slope"}

        # -------------------------------------------------------------
        # 2) Initialize data structures for coverage & CI lengths
        # -------------------------------------------------------------
        coverage_data_nonconservative = (
            {}
        )  # coverage_data[d][method] = list of bools across seeds
        lengths_data = (
            {}
        )  # lengths_data[d][method] = list of floats (CI widths) across seeds
        coverage_data_conservative = (
            {}
        )  # coverage_data[d][method] = list of bools across seeds

        for d in dims_list:
            coverage_data_nonconservative[d] = {}
            lengths_data[d] = {}
            coverage_data_conservative[d] = {}
            for method in all_methods:
                coverage_data_nonconservative[d][method] = []
                lengths_data[d][method] = []
                coverage_data_conservative[d][method] = []

        # -------------------------------------------------------------
        # 3) Loop over each seed's result, gather coverage booleans & CI lengths
        # -------------------------------------------------------------
        for seed_val, seed_result in self.results.items():
            # seed_result => {method_name -> {dim_str -> {...}}}
            for method in all_methods:
                dim_dict = seed_result[method]
                for d in dims_list:
                    entry = dim_dict[str(d)]
                    covered_nonconservative = entry.get("ci_contains_param", False)
                    covered_conservative = entry.get("ci_diff_contains_zero", False)
                    ci = entry.get("ci", (0, 0))  # (lower, upper)
                    ci_length = ci[1] - ci[0]
                    coverage_data_nonconservative[d][method].append(
                        covered_nonconservative
                    )
                    lengths_data[d][method].append(ci_length)
                    coverage_data_conservative[d][method].append(covered_conservative)

        # -------------------------------------------------------------
        # 4) Convert to fraction coverage & average CI length
        # -------------------------------------------------------------

        coverage_fraction_nonconservative = {}
        average_length = {}
        coverage_fraction_conservative = {}

        for d in dims_list:
            coverage_fraction_nonconservative[d] = {}
            average_length[d] = {}
            coverage_fraction_conservative[d] = {}
            for method in all_methods:
                coverage_fraction_nonconservative[d][method] = np.mean(
                    coverage_data_nonconservative[d][method]
                )
                average_length[d][method] = np.mean(lengths_data[d][method])
                coverage_fraction_conservative[d][method] = np.mean(
                    coverage_data_conservative[d][method]
                )

        # reorder the methods for plotting to have "Ours" last
        all_methods.remove("Ours")
        all_methods.append("Ours")

        # -------------------------------------------------------------
        # 5) Create subplots: 2 or 3 rows depending on conservative bool, len(dims_list) columns
        #    First row = conservative coverage, second row = average length, (optional) third row = both coverages
        # -------------------------------------------------------------
        n_dims = len(dims_list)
        if conservative:
            fig, axs = plt.subplots(3, n_dims, figsize=(4.0 * n_dims, 9), sharex=True)
        else:
            fig, axs = plt.subplots(2, n_dims, figsize=(4.0 * n_dims, 5), sharex=True)

        # Helper to handle indexing when n_dims=1
        def get_ax(axs, dim_idx, row=0):
            if n_dims == 1:
                return axs[row]
            else:
                return axs[row, dim_idx]

        x_positions = np.arange(len(all_methods))
        # pick one color for our method and one for the others color-blind friendly (e.g. blue vs grey)
        colors = {"Ours": "#1f77b4", "Others": "#999999"}

        # -------------------------------------------------------------
        # 6) Top row: conservative coverage bar charts
        # -------------------------------------------------------------
        for col_idx, d in enumerate(dims_list):
            ax_cov = get_ax(axs, col_idx, row=0)
            coverages_for_this_dim = coverage_fraction_conservative[d]

            for i, method in enumerate(all_methods):
                frac = coverages_for_this_dim[method]
                if method == "Ours":
                    color_plot = colors["Ours"]
                else:
                    color_plot = colors["Others"]
                ax_cov.bar(
                    x_positions[i],
                    frac,
                    color=color_plot,
                    label=method if col_idx == 0 else "",  # label only once
                )

                print(f"Method: {method}, Dim: {d}, Coverage: {frac}")

            # Dashed horizontal line at nominal coverage
            ax_cov.axhline(
                1 - self.alpha,
                color="black",
                linestyle="--",
                label=("Nominal coverage" if col_idx == 0 else ""),
            )

            ax_cov.set_ylim(0, 1)
            ax_cov.set_xticks(x_positions)
            ax_cov.set_xticklabels(all_methods, rotation=45, ha="right", fontsize=fontsize-5)
            ax_cov.set_yticklabels(ax_cov.get_yticks(), fontsize=fontsize-5)


            # Get a descriptive title from dim_title_map:
            dim_int = int(d)
            dim_label = dim_title_map.get(dim_int, f"Dimension {d}")
            ax_cov.set_title(dim_label, fontsize=fontsize)

            if col_idx == 0:
                ax_cov.set_ylabel("Coverage", fontsize=fontsize-5)
            else:
                ax_cov.set_ylabel("", fontsize=fontsize-5)

        # -------------------------------------------------------------
        # 7) Bottom row: average CI length bar charts
        # -------------------------------------------------------------
        for col_idx, d in enumerate(dims_list):

            if conservative:
                ax_len = get_ax(axs, col_idx, row=2)
            else:
                ax_len = get_ax(axs, col_idx, row=1)

            lengths_for_this_dim = average_length[d]

            for i, method in enumerate(all_methods):
                avg_len = lengths_for_this_dim[method]
                if method == "Ours":
                    color_plot = colors["Ours"]
                else:
                    color_plot = colors["Others"]
                ax_len.bar(
                    x_positions[i],
                    avg_len,
                    color=color_plot,
                    label=method if col_idx == 0 else "",  # label only once
                )

            ax_len.set_xticks(x_positions)
            ax_len.set_xticklabels(all_methods, rotation=45, ha="right", fontsize=fontsize-5)
            ax_len.set_yticklabels(ax_len.get_yticks(), fontsize=fontsize-5)


            if col_idx == 0:
                ax_len.set_ylabel("Average CI Width", fontsize=fontsize-5)
            else:
                ax_len.set_ylabel("", fontsize=fontsize-5)

        # -------------------------------------------------------------
        # (8) (Optional) Very bottom row: conservative coverage bar charts
        # -------------------------------------------------------------
        if conservative:
            for col_idx, d in enumerate(dims_list):
                ax_cov = get_ax(axs, col_idx, row=1)
                coverages_for_this_dim = coverage_fraction_nonconservative[d]

                for i, method in enumerate(all_methods):
                    frac = coverages_for_this_dim[method]
                    if method == "Ours":
                        color_plot = colors["Ours"]
                    else:
                        color_plot = colors["Others"]
                    ax_cov.bar(
                        x_positions[i],
                        frac,
                        color=color_plot,
                        label=method if col_idx == 0 else "",  # label only once
                    )

                # Dashed horizontal line at nominal coverage
                ax_cov.axhline(
                    1 - self.alpha,
                    color="black",
                    linestyle="--",
                    label=("Nominal coverage" if col_idx == 0 else ""),
                )

                ax_cov.set_ylim(0, 1)
                ax_cov.set_xticks(x_positions)
                ax_cov.set_xticklabels(all_methods, rotation=45, ha="right", fontsize=fontsize-5)
                ax_cov.set_yticklabels(ax_cov.get_yticks(), fontsize=fontsize-5)


                # Get a descriptive title from dim_title_map:
                dim_int = int(d)
                dim_label = dim_title_map.get(dim_int, f"Dimension {d}")
                ax_cov.set_title(dim_label, fontsize=fontsize)

                if col_idx == 0:
                    ax_cov.set_ylabel("Coverage (of point estimate)", fontsize=fontsize-5)
                else:
                    ax_cov.set_ylabel("", fontsize=fontsize-5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)
        fname = (
            "coverages_by_dim_conservative.pdf"
            if conservative
            else "coverages_by_dim.pdf"
        )

        plt.savefig(Path(self.results_dir, fname))
        plt.close()

    def plot_confidence_intervals(self, filename="confidence_interval_plot.pdf"):
        """
        Creates a forest-plot-like figure of the CIs for each method,
        with seeds on rows and dimensions on columns.

        We assume `self.results` is saved as JSON with keys = seed strings ('0','1','2',...),
        and each seed => {method_name => {dim_str => {...}}}.
        """
        if self.results is None:
            print("No results to plot.")
            return

        # Set font size and latex format
        fontsize = 20
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("font", size=fontsize)

        # If self.results is a path or not loaded, load it
        if isinstance(self.results, str) or not isinstance(self.results, dict):
            self.results = json_tricks.load(str(Path(self.results_dir, "results.json")))

        # Set how many seeds & dims to plot
        seeds_to_plot = range(self.seeds_to_plot)
        all_dims_in_json = list(self.results["0"]["Ours"].keys())
        all_dims_in_json.sort(key=int)  # sort numerically
        dims_to_plot = [d for d in all_dims_in_json if d != "0"]

        n_seeds = len(seeds_to_plot)
        n_dims = len(dims_to_plot)

        if n_dims == 0 or n_seeds == 0:
            print("No valid dimensions or seeds to plot.")
            return

        # Dictionary for custom dimension names
        dim_title_map = {
            1: "Aridity Index",
            2: "Elevation",
            3: "Slope",
        }

        # ------------------------------------------------------------------------
        # 1) Precompute min_x and max_x for each dimension across all seeds & methods
        # ------------------------------------------------------------------------
        dim_ranges = {}
        for dim_key in dims_to_plot:
            min_x = float("inf")
            max_x = float("-inf")

            for seed_idx in seeds_to_plot:
                seed_result = self.results[str(seed_idx)]
                methods = list(seed_result.keys())

                for method in methods:
                    dim_info = seed_result[method][str(dim_key)]
                    lower, upper = dim_info["ci"]
                    # Update global min/max
                    if lower < min_x:
                        min_x = lower
                    if upper > max_x:
                        max_x = upper

                
                # Also account for true_params if available
                if seed_result['Ground Truth'] is not None:
                    true_val = seed_result['Ground Truth'][dim_key]['point_estimate']
                    if true_val < min_x:
                        min_x = true_val
                    if true_val > max_x:
                        max_x = true_val
                    # Possibly also account for confidence interval
                if seed_result['Ground Truth'] is not None:
                    ci_low, ci_high = seed_result['Ground Truth'][dim_key]['ci'][0], seed_result['Ground Truth'][dim_key]['ci'][1]
                    if ci_low < min_x:
                        min_x = ci_low
                    if ci_high > max_x:
                        max_x = ci_high

            # Store the computed range
            dim_ranges[dim_key] = (min_x, max_x)

        # ------------------------------------------------------------------------
        # 2) Create subplots with sharex='col' so each column shares the same x-limits
        # ------------------------------------------------------------------------
        fig, axs = plt.subplots(
            nrows=n_seeds,
            ncols=n_dims,
            figsize=(4.5 * n_dims, 2.5 * n_seeds),
            sharex="col",  # share x-axis within each column
            sharey=True,
        )

        # Helper to handle 1D cases
        def get_subplot(axs, row, col):
            if n_seeds > 1 and n_dims > 1:
                return axs[row, col]
            elif n_seeds > 1 and n_dims == 1:
                return axs[row]
            elif n_seeds == 1 and n_dims > 1:
                return axs[col]
            else:
                return axs  # only 1 row & 1 column

        # ------------------------------------------------------------------------
        # 3) Plot each seed (row) x dimension (column)
        # ------------------------------------------------------------------------
        for row_idx, seed_idx in enumerate(seeds_to_plot):
            for col_idx, dim_key in enumerate(dims_to_plot):
                ax = get_subplot(axs, row_idx, col_idx)
                seed_result = self.results[str(seed_idx)]
                methods = list(seed_result.keys())
                # drop ground truth
                methods = [method for method in methods if method != "Ground Truth"]
                y_positions = range(len(methods))

                # Plot each method's CI
                for y, method in zip(y_positions, methods):
                    dim_info = seed_result[method][str(dim_key)]
                    lower, upper = dim_info["ci"]
                    point_est = dim_info["point_estimate"]

                    ax.hlines(y=y, xmin=lower, xmax=upper, color="C0", linewidth=2)
                    ax.plot(point_est, y, "o", color="C0")

                # Plot "true" lines if available
                if seed_result['Ground Truth'][dim_key]['point_estimate'] is not None:
                    ax.axvline(
                            seed_result['Ground Truth'][dim_key]['point_estimate'], color="black", linestyle="--"
                    )
                    if seed_result['Ground Truth'][dim_key]['ci'] is not None:
                        ci_low, ci_high = seed_result['Ground Truth'][dim_key]['ci'][0], seed_result['Ground Truth'][dim_key]['ci'][1]
                        ax.axvline(ci_low, color="orange", linestyle="--")
                        ax.axvline(ci_high, color="orange", linestyle="--")

                # Y-ticks show the method names
                ax.set_yticks(list(y_positions), )
                ax.set_yticklabels(methods, fontsize=fontsize-5)

                # Label left-most column with "Seed X"
                if col_idx == 0:
                    ax.set_ylabel(f"Seed {seed_idx}", fontsize=fontsize-5)
                else:
                    ax.set_ylabel("", fontsize=fontsize-5)

                # Label top row with dimension name
                if row_idx == 0:
                    dim_int = int(dim_key)
                    dim_label = dim_title_map.get(dim_int, f"Dimension {dim_key}")
                    ax.set_title(dim_label, fontsize=fontsize-5)

        # ------------------------------------------------------------------------
        # 4) Set x-limits on the top subplot in each column (which shares with the column)
        # ------------------------------------------------------------------------
        for col_idx, dim_key in enumerate(dims_to_plot):
            top_ax = (
                get_subplot(axs, 0, col_idx)
                if n_seeds > 1
                else get_subplot(axs, 0, col_idx)
            )
            min_x, max_x = dim_ranges[dim_key]
            padding = 0.05 * (max_x - min_x)
            top_ax.set_xlim(min_x - padding, max_x + padding)

        plt.tight_layout()
        plt.savefig(Path(self.results_dir, filename))
        plt.close()
