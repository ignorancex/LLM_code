import dill as pickle
import os
import os
from pathlib import Path
from certifiable_uda_loc.run_monte_carlo import main_monte_carlo
from certifiable_uda_loc.trials import MonteCarloSummary
from matplotlib import pyplot as plt
import seaborn as sns
from collections import namedtuple
from certifiable_uda_loc.parsing import ParserArguments
from typing import Dict, Union
import itertools
from certifiable_uda_loc.run_monte_carlo import MonteCarloResultComparison

import numpy as np
import certifiable_uda_loc.generate_se2_cases as gen_se2
import certifiable_uda_loc.subtask_factors as subtask_factors
import certifiable_uda_loc.test_cases as test_cases

import certifiable_uda_loc.utils.conversion_utils as conversion_utils
import navlie.utils.plot as nv_plot
import certifiable_uda_loc.path_configuration as path_config


def main():
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("colorblind"), n_colors=5)
    stylesheet_path = os.path.join(
        path_config.project_root_dir, "stylesheet_wide.mplstyle"
    )
    plt.style.use(stylesheet_path)
    top_level_result_dir = os.path.join(path_config.top_level_result_dir, "liw_paper")
    # run_id = "tmin_overall_liw-0_n_poses-5_n_landmarks-2_landmarks_to_use-None_max_num_subsequences-50_pose_spacing-20_n_jobs-3_use_ground_truth_for_relative_pose_liw-False_use_ground_truth_for_relative_landmark_meas_liw-False"
    # mc_run_number = 2

    run_id = "tmin_overall_liw-0_n_poses-5_n_landmarks-3_landmarks_to_use-None_max_num_subsequences-50_pose_spacing-20_n_jobs-3_use_ground_truth_for_relative_pose_liw-False_use_ground_truth_for_relative_landmark_meas_liw-False"
    mc_run_number = 4

    results_dir = os.path.join(
        top_level_result_dir,
        run_id,
    )

    # If running this from the repository where all the results have not been generated yet.
    # The specific result for Fig1 in our paper is the example_result dir. To use it, keep following line uncommented.
    results_dir = os.path.join(path_config.project_root_dir, "example_result")

    fig_dir = os.path.join(path_config.top_level_fig_dir, "specific_run")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    # labels_to_exclude = ["DR Optimization", "GT Associations", "DR", "GT"]
    labels_to_exclude = ["DR Optimization", "DR", "Max-Mix GT Init"]
    labels_to_exclude = ["DR Optimization", "Max-Mix GT Init", "GT Associations"]
    label_to_legend_label_dict = {
        "SDP": "Proposed",
        "Max-Mix GT Init": "Local method init. at ground truth",
        "Max-Mix DR Init": "Local method init. at dead-reckoned",
        "DR": "Dead-reckoned",
        "GT": "Ground truth",
        "GT Associations": "Ground truth associations",
    }
    with open(
        os.path.join(results_dir, f"mc_result_comparison_{mc_run_number}.pkl"),
        "rb",
    ) as f:
        mc_result_comparison: MonteCarloResultComparison = pickle.load(f)
        mc_result_comparison.set_solutions_dict()
        mc_result_comparison.set_data_associations_dict()
        # Question: How tso plot measurements?
        # mc_result has the measurements in it...

        landmark_list = conversion_utils.extract_unique_landmarks(
            mc_result_comparison.mc_result_sdp.fg
        )
        # colors = get_plot_colormap(10)
        colors = sns.color_palette("colorblind")
        # colors = colors[0:4]
        colors = [colors[0], colors[1], colors[2], colors[7]]

        linestyles = ["-", "--", ":", ":", ":"]
        lv_style = 0
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        for lv1, (label, se2_states) in enumerate(
            mc_result_comparison.solutions_dict.items()
        ):

            if label in labels_to_exclude:
                continue
            print(lv_style)

            fig, ax = nv_plot.plot_poses(
                se2_states,
                ax,
                triad_color=colors[lv_style],
                arrow_length=0.03,
                step=1,
                label=label_to_legend_label_dict[label],
                kwargs_line={
                    "linestyle": linestyles[lv_style],
                    "color": colors[lv_style],
                },
                axes_equal=False,
            )
            lv_style += 1
            if label == "GT":
                for lv_state, state in enumerate(se2_states):
                    r = state.position
                    plt.text(r[0], r[1] + 0.1, f"T{lv_state}", size=14)

        for l in landmark_list:
            ax.scatter(l[0], l[1], color="blue")

    fig_path = os.path.join(fig_dir, f"mc_result_comparison_{mc_run_number}.pdf")
    # ax.set_xlim([-6, 2])
    # ax.set_ylim([-3, 6])
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.ylim([-2, 2])
    # plt.legend(loc="upper left")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
    )
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved at Fig path: {fig_path}")
    print("Landmark list")
    landmark_string = ""
    for l in landmark_list:
        landmark_string += f"({l[0]:.1f},{l[1]:.1f}),"
    print(landmark_string)
    # array([5.58689828, 1.40386939]), array([1.98101489, 8.00744569]), array([9.68261576, 3.13424178])]


if __name__ == "__main__":

    main()
