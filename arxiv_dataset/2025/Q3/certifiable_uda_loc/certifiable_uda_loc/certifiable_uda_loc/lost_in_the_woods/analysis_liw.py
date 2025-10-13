from tqdm import tqdm
from typing import Tuple
import dill as pickle
from typing import Dict, List, Any
import itertools
import os
from certifiable_uda_loc.parsing import ParserArguments
import pandas as pd
import os
from pathlib import Path
from loguru import logger
from certifiable_uda_loc.run_monte_carlo import main_monte_carlo
from typing import List
import dill as pickle
from certifiable_uda_loc.trials import MonteCarloSummary
from matplotlib import pyplot as plt
import seaborn as sns
from collections import namedtuple
from certifiable_uda_loc.parsing import ParserArguments
from typing import Dict, Union
import itertools
from certifiable_uda_loc.run_monte_carlo import MonteCarloResultComparison
from pathlib import Path
import certifiable_uda_loc.lost_in_the_woods.convert_to_fg as liw_convert_to_fg
from typing import List
import certifiable_uda_loc.path_configuration as path_config


def default_arg_dict_liw():
    arg_dict = {
        "run_id": "test",
        "verbose": False,
        "results_dir": path_config.top_level_result_dir,
        "n_trials_per_run": 1,
        "create_constraints_from_nullspace": False,
        "create_discrete_variable_constraints_from_nullspace": False,
        "discrete_variable_constraints": ["bool", "prod_ci_cj", "sum_one"],
        "n_jobs": 1,
        "use_sparse_matrices_properly": True,
        "sparse_bool_cont_variables": True,
        "no_sparsify_interfactor_discrete_constraints": False,
        "no_sparsify_A_th_premultiplication": False,
        "no_sparsify_cx_premultiplication_of_A_th": False,
        "no_sparsify_cxi_cxj_premultiplication_of_A_th": False,
        "no_sparsify_analytic_moment_constraints": False,
        "no_sparsify_analytic_moment_constraints_2": False,
        "no_sparsify_analytic_moment_constraints_3": False,
        "no_sparsify_off_diag_boolean": True,
        "no_sparsify_bool_product_moment": False,
        "problem_type": "se2",
        "subtask": "localization",
        "prior_rot_kappa_langevin_inverse": 0.01,
        "prior_pos_cov": 0.01,
        "prior_landmark_cov": 200,
        "prior_landmark_noise_corrupt_cov": None,
        "rel_rot_kappa_langevin_inverse_base_val": 0.01,
        "rel_pos_base_val": 0.1,
        "rel_noise_scale": 1,
        "rel_landmark_meas_cov": 0.01,
        "n_poses": 3,
        "n_landmarks": 2,
        "experiment_type": "lost_in_the_woods",  # simulation or lost_in_the_woods
        "overall_time_bounds": [200, 210],  # only applicable for lost_in_the_woods
        "landmark_spread": 10,
        "meas_per_landmark": 1,
        "fraction_removal": 0,
        "uda_fraction": 1,
        "noiseless": False,
        "locked_first_pose": False,
        "solver": "mosek",
        "solver_primal": False,
        "solver_cost_matrix_adjust": True,
        "prior_landmark_location": None,
    }
    return arg_dict


def string_id_from_parameters(param_dict: Dict[str, Union[float, str]]):
    string_id = ""
    for param_name, val in param_dict.items():
        string_id += f"{param_name}-{val}_"
    string_id = string_id[:-1]
    return string_id


def parameters_from_string_id(string_id: str):
    string_list = string_id.split("_")
    param_dict = {}
    for param_string in string_list:
        param_list = param_string.split("-")
        param_name = param_list[0]
        param_list = param_string.split("-")
        val = [1]
        try:
            val = float(val)
        except ValueError:
            pass
        param_dict[param_name] = val


from certifiable_uda_loc.analysis_machinery import (
    Analysis,
    comparison_heatmaps,
    comparison_heatmap_three_plots,
)
import numpy as np
import certifiable_uda_loc.generate_se2_cases as gen_se2
import certifiable_uda_loc.subtask_factors as subtask_factors
import certifiable_uda_loc.test_cases as test_cases
import certifiable_uda_loc.lost_in_the_woods.dataset as dataset
from certifiable_uda_loc.lost_in_the_woods import noise_properties
import certifiable_uda_loc.path_configuration as path_config


class AnalysisLIW(Analysis):
    def __init__(
        self,
        analysis_id: str,
        parameter_dict_list: List[Dict[str, Any]],
        param_settings_list_breakdown: List[Dict[str, Any]],
        root_dir: str,
        liw_ds_path: str = path_config.lost_in_the_woods_dataset_path,
        default_arg_dict: Dict = None,
    ):
        self.analysis_id = analysis_id
        self.parameter_dict_list = parameter_dict_list
        self.param_settings_list_breakdown = param_settings_list_breakdown
        self.root_dir = root_dir
        self.top_level_fig_dir: str = os.path.join(self.root_dir, "figs", analysis_id)
        self.top_level_result_dir: str = os.path.join(
            self.root_dir, "mc_result", analysis_id
        )
        self.top_level_factor_graph_dir: str = os.path.join(
            root_dir, "factor_graphs", analysis_id
        )
        self.liw_ds_path = liw_ds_path
        self.default_arg_dict = default_arg_dict

    def compute_noise_values(self, n_poses: int, pose_spacing: int, fname_save: str):
        subseq_len = n_poses * pose_spacing
        tmin_noise = 0
        tmax_noise = 1200
        n_poses_noise = n_poses
        liw_ds: dataset.LostInTheWoodsDataset = (
            dataset.LostInTheWoodsDataset.from_mat_file(
                self.liw_ds_path,
                time_bounds=[tmin_noise, tmax_noise],
            )
        )
        cov_lndmrk_meas = noise_properties.translation_from_range_bearing_noise(
            liw_ds, [tmin_noise, tmax_noise], n_poses_noise
        )
        kappa, loc, cov_dr = noise_properties.relative_pose_noise(
            liw_ds,
            [tmin_noise, tmax_noise],
            n_poses_noise,
            von_mises_fig_path="./dtheta_hist.pdf",
        )
        if fname_save is not None:
            noise_dict = {
                "kappa": kappa,
                "cov_dr": cov_dr,
                "cov_lndmrk_meas": cov_lndmrk_meas,
            }
            with open(fname_save, "wb") as f:
                pickle.dump(noise_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return kappa, cov_dr, cov_lndmrk_meas

    # Do I want to specify the noise obtention parameters?
    def factor_graph_setup_from_args(
        self, args: ParserArguments, fnames: List[str], seed_override: int = None
    ):
        subseq_len = args.n_poses * args.pose_spacing
        # tmin_noise = 0
        # tmax_noise = 1200
        # n_poses_noise = args.n_poses
        # liw_ds: dataset.LostInTheWoodsDataset = (
        #     dataset.LostInTheWoodsDataset.from_mat_file(
        #         self.liw_ds_path,
        #         time_bounds=[tmin_noise, tmax_noise],
        #     )
        # )
        # cov_lndmrk_meas = noise_properties.translation_from_range_bearing_noise(
        #     liw_ds, [tmin_noise, tmax_noise], n_poses_noise
        # )
        # kappa, loc, cov_dr = noise_properties.relative_pose_noise(
        #     liw_ds,
        #     [tmin_noise, tmax_noise],
        #     n_poses_noise,
        #     von_mises_fig_path="./dtheta_hist.pdf",
        # )
        kappa, cov_dr, cov_lndmrk_meas = self.compute_noise_values(
            args.n_poses, args.pose_spacing, None
        )

        for lv_seed in range(len(fnames)):
            fg_path = fnames[lv_seed]
            tmin = args.tmin_overall_liw + subseq_len * lv_seed
            tmax = tmin + args.n_poses * args.pose_spacing

            liw_ds: dataset.LostInTheWoodsDataset = (
                dataset.LostInTheWoodsDataset.from_mat_file(
                    self.liw_ds_path,
                    time_bounds=[tmin, tmax],
                )
            )
            noise_parameters = gen_se2.Se2CaseNoiseParameters(
                prior_rot_kappa_langevin_inverse=args.prior_rot_kappa_langevin_inverse,
                prior_pos_cov=args.prior_pos_cov,
                prior_landmark_cov=None,
                prior_landmark_noise_corrupt_cov=None,
                rel_rot_kappa_langevin_inverse=1 / kappa,
                rel_pos_cov=np.mean(np.diag(cov_dr)),
                rel_landmark_meas_cov=np.mean(np.diag(cov_lndmrk_meas)),
            )
            fg = liw_convert_to_fg.get_factor_graph_from_liw(
                liw_ds=liw_ds,
                time_bounds=[tmin, tmax],
                n_poses=args.n_poses,
                noise_parameters=noise_parameters,
                landmark_indices_to_use=args.landmarks_to_use,
                num_landmarks_to_use=args.n_landmarks,
                unknown_data_association=args.unknown_data_association_liw,
                use_ground_truth_for_relative_pose=args.use_ground_truth_for_relative_pose_liw,
                use_ground_truth_for_relative_landmark_meas=args.use_ground_truth_for_relative_landmark_meas_liw,
            )

            # print(fg_path)
            with open(
                fg_path,
                "wb",
            ) as f:
                pickle.dump(fg, f)

    def plot_df(
        self,
        df: pd.DataFrame,
        fig_dir,
        xval="rel_landmark_meas_cov",
        yval="rel_landmark_meas_cov",
        kwargs_heatmap={},
    ):
        label_n_poses = "Num. Poses"
        label_pose_spacing = "Pose Spacing (s)"
        params = {
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "text.usetex": True,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
        plt.rcParams.update(params)

        sns.set_style("whitegrid")
        palette = sns.color_palette("colorblind")
        """-----------TIGHTNESS - MEDIAN OF EIGENVALUE RATIO-----------"""
        df2 = df.pivot_table(
            index=xval,
            columns=yval,
            values="log_eigval_ratio",
            aggfunc="median",
        )
        df2 = df2.sort_index(ascending=False)
        plt.figure()
        sns.heatmap(
            df2,
            annot=True,
            cmap="coolwarm",
            cbar_kws={"label": "median log eigval ratio"},
        )
        plt.savefig(os.path.join(fig_dir, "log_eigval_ratio_median.pdf"))

        params = {
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "text.usetex": True,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
        plt.rcParams.update(params)
        """-----------POSITION ERRORS-----------"""

        for tight, tight_label in zip(
            [True, False, None], ["tight", "nontight", "all"]
        ):

            fig, axs = comparison_heatmaps(
                df,
                xval=xval,
                yval=yval,
                values=[
                    "sdp_error_wrt_mm_gt_init_pos",
                    "dr_error_wrt_mm_gt_init_pos",
                    # "mm_sdp_init_error_wrt_mm_gt_init_pos",
                ],
                aggfunc="median",
                tight_cases=tight,
                label="Median Position Error Across Trials",
                kwargs_plot=dict(
                    {
                        "vmin": 0,
                        "vmax": 0.5,
                        "cmap": "coolwarm",
                        "annot_kws": {"fontsize": 14},
                        "fmt": ".1e",
                    },
                    **kwargs_heatmap,
                ),
            )
            # ax1: plt.Axes = ax1
            # ax2: plt.Axes = ax2
            axs[0].set_title("SDP")
            axs[1].set_title("MM DR INIT")
            axs[1].set_ylabel(None)
            axs[0].set_ylabel(label_pose_spacing)
            axs[0].set_xlabel(label_n_poses)
            axs[1].set_xlabel(label_n_poses)
            plt.savefig(os.path.join(fig_dir, f"pos_errors_vs_noise_{tight_label}.pdf"))

        """-----------ROTATION ERRORS-----------"""
        for tight, tight_label in zip([True, False], ["tight", "nontight"]):
            fig, axs = comparison_heatmaps(
                df,
                xval=xval,
                yval=yval,
                values=[
                    "sdp_error_wrt_mm_gt_init_rot",
                    "dr_error_wrt_mm_gt_init_rot",
                    # "mm_sdp_init_error_wrt_mm_gt_init_rot",
                ],
                aggfunc="median",
                label="Median Rot Error w.r.t. MM Init at GT",
                tight_cases=tight,
                kwargs_plot=dict(
                    {
                        "vmin": 0,
                        "vmax": 0.3,
                        "cmap": "coolwarm",
                        "annot_kws": {"fontsize": 10},
                        "fmt": ".1e",
                    },
                    **kwargs_heatmap,
                ),
            )

            axs[0].set_title("SDP")
            axs[1].set_title("MM DR INIT")
            # axs[2].set_title("MM SDP INIT")
            plt.savefig(os.path.join(fig_dir, f"rot_errors_vs_noise_{tight_label}.pdf"))

        for tight, tightlabel in zip([True, False, None], ["tight", "nontight", "all"]):

            plt.rcParams.update(params)
            fig, axs = comparison_heatmap_three_plots(
                df,
                xval=xval,
                yval=yval,
                values=[
                    "sdp_da_wrong",
                    "mm_dr_init_da_wrong",
                    "mm_gt_init_da_wrong",
                ],
                aggfunc="mean",
                tight_cases=tight,
                label="Data Assoc. Error Rate",
                kwargs_plot=dict(
                    {
                        "vmin": 0,
                        "vmax": 0.7,
                        "cmap": "coolwarm",
                        "annot_kws": {"fontsize": 18},
                        "fmt": ".2f",
                    },
                    **kwargs_heatmap,
                ),
            )

            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            for ax in axs:
                ax.set_xlabel(label_n_poses)
                ax.set_ylabel(label_pose_spacing)
            # ax1.set_aspect("equal")
            # ax2.set_aspect("equal")
            # ax3.set_aspect("equal")
            ax1.set_title("SDP")
            ax2.set_title("MM DR INIT")
            ax3.set_title("MM GT INIT")
            ax2.set_ylabel(None)
            ax3.set_ylabel(None)
            ax1.set_ylabel("Pose Spacing (s)")
            for ax in axs:
                ax.set_xlabel("Num. Poses")
            plt.savefig(os.path.join(fig_dir, f"data_associations_{tightlabel}.pdf"))

            # Next breakdown. For cases where data associations are NOT the same...
            # Plot percentage of cases where SDP cost < Max Mix GT Cost.
            df["sdp_minus_gt_association_cost"] = (
                df["cost_sdp"] - df["cost_gt_association"]
            )
            df1 = df[df["sdp_da_wrong_count"] != df["gt_association_da_wrong_count"]]

            df_val = df1[df1["is_tight"] == tight].pivot_table(
                index=xval,
                columns=yval,
                values="sdp_minus_gt_association_cost",
                aggfunc="max",
            )
            df_val = df_val.sort_index(ascending=False)

            if not df_val.empty:
                plt.figure()
                sns.heatmap(df_val, cbar=False, cmap="coolwarm", **kwargs_heatmap)
                plt.savefig(
                    os.path.join(
                        fig_dir, f"sdp_minus_gt_association_cost_{tightlabel}_max.pdf"
                    )
                )
            if not df_val.empty:
                plt.figure()
                df_val = df[df["is_tight"] == tight].pivot_table(
                    index=xval,
                    columns=yval,
                    values="sdp_minus_gt_association_cost",
                    aggfunc="mean",
                )
                df_val = df_val.sort_index(ascending=False)
                plt.figure()
                sns.heatmap(df_val, cbar=False, cmap="coolwarm", **kwargs_heatmap)
                plt.savefig(
                    os.path.join(
                        fig_dir, f"sdp_minus_gt_association_cost_{tightlabel}_mean.pdf"
                    )
                )

        df2 = df[df["mm_gt_init_da_wrong"] != df["sdp_da_wrong_count"]].pivot_table(
            index=xval,
            columns=yval,
            values="sdp_minus_gt_association_cost",
            aggfunc="mean",
        )
        if not df2.empty:
            plt.figure()
            sns.heatmap(df2, cbar=False, cmap="coolwarm", **kwargs_heatmap)
            plt.savefig(os.path.join(fig_dir, f"sdp_da_not_gt_da_cost_diff.pdf"))

        df_val = df.pivot_table(
            index=xval,
            columns=yval,
            values="is_tight",
            aggfunc="mean",
        )
        df_val = df_val.sort_index(ascending=False)
        plt.figure()
        sns.heatmap(
            df_val,
            cbar=True,
            cmap=sns.cubehelix_palette(as_cmap=True),
            annot_kws={"fontsize": 15},
            **kwargs_heatmap,
        )
        ax: plt.Axes = plt.gca()
        ax.set_xlabel(label_n_poses)
        ax.set_ylabel(label_pose_spacing)
        plt.savefig(os.path.join(fig_dir, f"tightness_fraction.pdf"))

        df_long = df_val.reset_index().melt(
            id_vars=xval, var_name=yval, value_name="mean_is_tight"
        )
        # df_long = df_val.reset_index().melt(id_vars=yval, var_name=xval, value_name="mean_is_tight")
        plt.figure(figsize=(10, 8))
        sns.barplot(x=xval, y="mean_is_tight", hue=yval, data=df_long, palette=palette)
        # Labels and title
        plt.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

        plt.xlabel("Relative pose measurement noise multiplier")
        plt.ylabel("Fraction of tight cases")
        plt.legend(
            title="Num. Poses", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5
        )
        # plt.legend(title=yval, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

        # Save the plot
        plt.savefig(os.path.join(fig_dir, "tightness_fraction_barchart.pdf"))


def get_analyses() -> Dict[str, AnalysisLIW]:
    return {
        "liw_paper": AnalysisLIW(
            analysis_id="liw_paper",
            parameter_dict_list=[
                {
                    "tmin_overall_liw": 0,
                    "n_poses": n_poses,
                    "n_landmarks": n_landmarks,
                    "landmarks_to_use": None,
                    "num_subsequences": 3,
                    "pose_spacing": pose_spacing,
                    "n_jobs": 3,
                    "use_ground_truth_for_relative_pose_liw": use_ground_truth_for_relative_pose_liw,
                    "use_ground_truth_for_relative_landmark_meas_liw": use_ground_truth_for_relative_landmark_meas_liw,
                    # num_subsequences: int = None
                    # The number of poses in each subsection is specified using n_poses
                    # pose_spacing: float = None  # Amount of time between each pose
                    # landmarks_to_use: List[int] = None  # Landmark ids to use for lost in the woods
                }
                for n_poses, n_landmarks, pose_spacing, use_ground_truth_for_relative_pose_liw, use_ground_truth_for_relative_landmark_meas_liw in itertools.product(
                    [3, 5], [2, 3], [20, 40, 60], [False], [False]
                )
            ],
            param_settings_list_breakdown=[
                {"use_ground_truth_for_relative_pose_liw": False}
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict_liw(),
        ),
        "liw_debug": AnalysisLIW(
            analysis_id="liw_debug",
            parameter_dict_list=[
                {
                    "tmin_overall_liw": 0,
                    "n_poses": n_poses,
                    "n_landmarks": n_landmarks,
                    "landmarks_to_use": None,
                    "max_num_subsequences": 30,
                    "pose_spacing": pose_spacing,
                    "n_jobs": 3,
                    "use_ground_truth_for_relative_pose_liw": use_ground_truth_for_relative_pose_liw,
                    "use_ground_truth_for_relative_landmark_meas_liw": use_ground_truth_for_relative_landmark_meas_liw,
                    # num_subsequences: int = None
                    # The number of poses in each subsection is specified using n_poses
                    # pose_spacing: float = None  # Amount of time between each pose
                    # landmarks_to_use: List[int] = None  # Landmark ids to use for lost in the woods
                }
                for n_poses, n_landmarks, pose_spacing, use_ground_truth_for_relative_pose_liw, use_ground_truth_for_relative_landmark_meas_liw in itertools.product(
                    [3], [2, 3], [20], [False], [False]
                )
            ],
            param_settings_list_breakdown=[
                {"use_ground_truth_for_relative_pose_liw": False}
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict_liw(),
        ),
        "liw_paper": AnalysisLIW(
            analysis_id="liw_paper",
            parameter_dict_list=[
                {
                    "tmin_overall_liw": 0,
                    "n_poses": n_poses,
                    "n_landmarks": n_landmarks,
                    "landmarks_to_use": None,
                    "max_num_subsequences": 30,
                    "pose_spacing": pose_spacing,
                    "n_jobs": 3,
                    "max_num_subsequences": 50,
                    "use_ground_truth_for_relative_pose_liw": use_ground_truth_for_relative_pose_liw,
                    "use_ground_truth_for_relative_landmark_meas_liw": use_ground_truth_for_relative_landmark_meas_liw,
                    # num_subsequences: int = None
                    # The number of poses in each subsection is specified using n_poses
                    # pose_spacing: float = None  # Amount of time between each pose
                    # landmarks_to_use: List[int] = None  # Landmark ids to use for lost in the woods
                }
                for n_poses, n_landmarks, pose_spacing, use_ground_truth_for_relative_pose_liw, use_ground_truth_for_relative_landmark_meas_liw in itertools.product(
                    [3, 5], [2, 3], [20, 40, 60], [False], [False]
                )
            ],
            param_settings_list_breakdown=[
                {"use_ground_truth_for_relative_pose_liw": False}
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict_liw(),
        ),
    }
