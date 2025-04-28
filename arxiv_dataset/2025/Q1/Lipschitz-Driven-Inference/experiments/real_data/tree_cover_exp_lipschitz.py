import os
import argparse
from pathlib import Path
import gpflow
from tree_cover_exp import TreeCoverExperiment
from lipschitz_driven_inference.estimators import (
    Estimator,
    NNLipschitzDrivenEstimator,
    Dataset,
)
from typing import Tuple, List, Dict, Optional
from numpy.typing import ArrayLike
import numpy as np
import json_tricks
import matplotlib.pyplot as plt


def parse():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=250)
    parser.add_argument("--num_neighbors", type=int, default=1)

    parser.add_argument(
        "--lipschitz_vals",
        nargs="+", 
        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1], 
        help="List of Lipschitz constants to try (space-separated)."
    )
    
    parser.add_argument("--num_threads", type=int, default=5)
    parser.add_argument("--seeds_to_plot", type=int, default=5)
    parser.add_argument("--region", type=str, default="south")
    return parser.parse_args()


class TreeCoverExperimentLipschitz(TreeCoverExperiment):
    """
    A variant of TreeCoverExperiment that only uses NNLinearEstimatorUnknownNoise 
    but tries multiple Lipschitz constants.
    """

    def run_single_simulation(self, seed: int) -> Dict[str, Dict]:
        """
        1) Generate data for a given seed (random subsample).
        2) Fit estimators, get point estimates, intervals, etc.
        """
        
        if seed % 50 == 0:
            print(f"Running seed {seed}")

        # Pass the seed so generate_data can do the random subsampling
        training_data, test_data = self.generate_data(seed=seed)
        dims = training_data.X.shape[1]
        # Compute the "true" parameter
        true_param = self.get_true_parameter(test_data)
        true_ci = self.get_true_ci(test_data)
        true_sd = self.get_true_sd(test_data)

        # Build the estimators and collect results
        results = {}
        for estimator, estimator_kwarg in zip(self.estimators, self.estimator_kwargs):
            estimator_kwarg_copy = estimator_kwarg.copy()

            estimator_instance = estimator(
                training_data, test_data, **estimator_kwarg_copy
            )
            L = estimator_kwarg_copy["lipschitz_bound"]
            for d in range(dims):
                point_estimate = estimator_instance.point_estimate(d)
                ci = estimator_instance.confidence_interval(d, self.alpha)
                sd = estimator_instance.sd_estimate(d)
                results[(L,d)] = {
                    "point_estimate": point_estimate,
                    "ci": ci,
                    "ci_contains_param": (ci.lower <= true_param[d] <= ci.upper),
                    "ci_overlaps_true_ci" : self.check_intervals_overlaps(ci.lower, ci.upper, true_ci[d][0], true_ci[d][1]),
                    "ci_diff_contains_zero" : self.check_interval_diff_contains_zero(d, estimator_instance, point_estimate, sd, true_param, true_sd)
                }
                
        return results

    def plot_coverage_results(self):
        """
        Produces one figure with subplots of shape (2 x n_dims):
        - The top row is coverage bar charts (one column per dimension)
        - The bottom row is average length bar charts (one column per dimension)
        """
        if self.results is None or len(self.results) == 0:
            print("No results to plot.")
            return

        # Import saved data if needed
        if isinstance(self.results, str) or not isinstance(self.results, dict):
            self.results = json_tricks.load(str(Path(self.results_dir, "results.json")))
        
        # -------------------------------------------------------------
        # 1) Gather dimension keys & methods
        # -------------------------------------------------------------
        # We skip the '0' dimension (intercept).
        lipschitz_constants = list(self.results['0'].keys())
        all_dims_in_json = list(self.results['0'][lipschitz_constants[0]].keys()) 
        all_dims_in_json.sort(key=int)  # numeric sort
        dims_list = [d for d in all_dims_in_json if d != '0']  # skip the intercept

        first_seed_result = self.results['0']  # => {'MethodA': {'0': {...}, '1': {...}}, ...}
        all_methods = list(first_seed_result.keys())
        all_methods.sort()

        # Define a lookup dict to rename dimension
        dim_title_map = {
            1: "Aridity Index",
            2: "Elevation",
            3: "Slope"
        }

        # -------------------------------------------------------------
        # 2) Initialize data structures for coverage & CI lengths
        # -------------------------------------------------------------
        coverage_data = {}  # coverage_data[d][method] = list of bools across seeds
        lengths_data = {}   # lengths_data[d][method] = list of floats (CI widths) across seeds
        conservative_coverage_data = {}  # same structure as coverage_data

        for d in dims_list:
            coverage_data[d] = {}
            lengths_data[d] = {}
            conservative_coverage_data[d] = {}
            for method in all_methods:
                coverage_data[d][method] = []
                lengths_data[d][method] = []
                conservative_coverage_data[d][method] = []

        # -------------------------------------------------------------
        # 3) Loop over each seed's result, gather coverage booleans & CI lengths
        # -------------------------------------------------------------
        for seed_val, seed_result in self.results.items():
            # seed_result => {method_name -> {dim_str -> {...}}}
            for method in all_methods:
                dim_dict = seed_result[method]
                for d in dims_list:
                    entry = dim_dict[str(d)]
                    covered = entry.get("ci_contains_param", False)
                    ci = entry.get("ci", (0, 0))  # (lower, upper)
                    ci_length = ci[1] - ci[0]
                    conservative_covered = entry.get("ci_diff_contains_zero", False)
                    coverage_data[d][method].append(covered)
                    lengths_data[d][method].append(ci_length)
                    conservative_coverage_data[d][method].append(conservative_covered)

        # -------------------------------------------------------------
        # 4) Convert to fraction coverage & average CI length
        # -------------------------------------------------------------
        coverage_fraction = {}
        average_length = {}
        conservative_coverage_fraction = {}
        for d in dims_list:
            coverage_fraction[d] = {}
            average_length[d] = {}
            conservative_coverage_fraction[d] = {}
            for method in all_methods:
                coverage_fraction[d][method] = np.mean(coverage_data[d][method])
                average_length[d][method] = np.mean(lengths_data[d][method])
                conservative_coverage_fraction[d][method] = np.mean(conservative_coverage_data[d][method])

        # -------------------------------------------------------------
        # 5) Create subplots: 3 rows, len(dims_list) columns
        #    First row = conservative coverage, second row = average length, third row = both coverage
        # -------------------------------------------------------------
        n_dims = len(dims_list)
        fig, axs = plt.subplots(3, n_dims, figsize=(4.0 * n_dims, 10), sharex=True)

        # Helper to handle indexing when n_dims=1
        def get_ax(axs, dim_idx, row=0):
            if n_dims == 1:
                return axs[row]
            else:
                return axs[row, dim_idx]

        x_positions = np.arange(len(all_methods))

        # -------------------------------------------------------------
        # 6) Top row: coverage bar charts
        # -------------------------------------------------------------
        for col_idx, d in enumerate(dims_list):
            ax_cov = get_ax(axs, col_idx, row=0)
            coverages_for_this_dim = coverage_fraction[d]

            for i, method in enumerate(all_methods):
                frac = coverages_for_this_dim[method]
                color_plot = 'gray'
                ax_cov.bar(
                    x_positions[i],
                    frac,
                    color=color_plot,
                    label=method if col_idx == 0 else ""  # label only once
                )

                print(f"Method: {method}, Dim: {d}, Coverage: {frac}")

            # Dashed horizontal line at nominal coverage
            ax_cov.axhline(
                1 - self.alpha,
                color="black",
                linestyle="--",
                label=("Nominal coverage" if col_idx == 0 else "")
            )

            ax_cov.set_ylim(0, 1)
            ax_cov.set_xticks(x_positions)
            ax_cov.set_xticklabels(all_methods, rotation=45, ha='right')

            # Get a descriptive title from dim_title_map:
            dim_int = int(d)  
            dim_label = dim_title_map.get(dim_int, f"Dimension {d}")
            ax_cov.set_title(dim_label)

            if col_idx == 0:
                ax_cov.set_ylabel("Coverage")
            else:
                ax_cov.set_ylabel("")

        # -------------------------------------------------------------
        # 6) Middle row: conservative coverage bar charts
        # -------------------------------------------------------------
        for col_idx, d in enumerate(dims_list):
            ax_cov = get_ax(axs, col_idx, row=1)
            coverages_for_this_dim = conservative_coverage_fraction[d]

            for i, method in enumerate(all_methods):
                frac = coverages_for_this_dim[method]
                color_plot = 'gray'
                ax_cov.bar(
                    x_positions[i],
                    frac,
                    color=color_plot,
                    label=method if col_idx == 0 else ""  # label only once
                )

            # Dashed horizontal line at nominal coverage
            ax_cov.axhline(
                1 - self.alpha,
                color="black",
                linestyle="--",
                label=("Nominal coverage" if col_idx == 0 else "")
            )

            ax_cov.set_ylim(0, 1)
            ax_cov.set_xticks(x_positions)
            ax_cov.set_xticklabels(all_methods, rotation=45, ha='right')

            # Get a descriptive title from dim_title_map:
            dim_int = int(d) 
            dim_label = dim_title_map.get(dim_int, f"Dimension {d}")
            ax_cov.set_title(dim_label)

            if col_idx == 0:
                ax_cov.set_ylabel("Coverage (conservative)")
            else:
                ax_cov.set_ylabel("")

        # -------------------------------------------------------------
        # 7) Bottom row: average CI length bar charts
        # -------------------------------------------------------------
        for col_idx, d in enumerate(dims_list):
            ax_len = get_ax(axs, col_idx, row=2)
            lengths_for_this_dim = average_length[d]

            for i, method in enumerate(all_methods):
                avg_len = lengths_for_this_dim[method]
                color_plot = 'gray'
                ax_len.bar(
                    x_positions[i],
                    avg_len,
                    color=color_plot,
                    label=method if col_idx == 0 else ""  # label only once
                )

            ax_len.set_xticks(x_positions)
            ax_len.set_xticklabels(all_methods, rotation=45, ha='right')

            if col_idx == 0:
                ax_len.set_ylabel("Average CI Width")
            else:
                ax_len.set_ylabel("")

        plt.tight_layout()
        # Shift layout to accommodate the legend at the top
        plt.subplots_adjust(top=0.80)

        fname = "coverages_by_dim.pdf"
        plt.savefig(Path(self.results_dir, fname))
        plt.close()
    
    

if __name__ == "__main__":
    args = parse()

    # Convert the lipschitz_vals strings to floats
    lipschitz_list = [float(x) for x in args.lipschitz_vals]
    
    file_path = Path(__file__).parent
    results_dir = f"results/real_data_lipschitz/{args.region}"
    results_dir = str(Path(file_path, results_dir))

    experiment = TreeCoverExperimentLipschitz(
        num_seeds=args.num_seeds,
        seeds_to_plot=args.seeds_to_plot,
        name="tree_cover_lipschitz",
        results_dir=results_dir,
        alpha=0.05,
        estimators=[NNLipschitzDrivenEstimator] * len(args.lipschitz_vals),
        estimator_kwargs=[
            {
                "num_neighbors": args.num_neighbors,
                "lipschitz_bound": lipbdd,
                "data_on_sphere": True,
            }
            for lipbdd in args.lipschitz_vals
        ],
        parallel_threads=args.num_threads,
        region=args.region,
    )

    # Run the experiment
    experiment.plot_data()
    experiment.run()
    experiment.save_results()
    # Additional plots
    experiment.plot_coverage_results()