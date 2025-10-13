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
import dill as pickle
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
from pathlib import Path
import certifiable_uda_loc.path_configuration as path_config

# Question: How to do analysis for lost in the woods with minimum code copypaste.
# The running of the analysis is similar to what we have below.
# But plotting and factor graph setup is differnet...


def default_arg_dict():
    arg_dict = {
        "run_id": "test",
        "verbose": False,
        "results_dir": path_config.top_level_result_dir,
        "n_trials_per_run": 1,
        "create_constraints_from_nullspace": False,
        "create_discrete_variable_constraints_from_nullspace": False,
        "discrete_variable_constraints": ["bool", "prod_ci_cj", "sum_one"],
        "n_jobs": 5,
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
        "experiment_type": "simulation",  # simulation or lost_in_the_woods
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


# def set_cmap_axes_parameters(axs: List[plt.Axes], labels: List[str]) -> None:
#     for ax in axs:
#         ax.xaxis.set_tick_params(labelsize=14)
#         ax.yaxis.set_tick_params(labelsize=14)
#         ax.set_xlabel(label_rel_pos_noise)
#         ax.set_ylabel(label_landmark_meas)
#         ax.xaxis.label.set_size(15)
#         ax.yaxis.label.set_size(15)


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


def comparison_heatmap_three_plots(
    df: pd.DataFrame,
    xval: str,
    yval: str,
    values: List[str],
    aggfunc: str,
    label: str,
    tight_cases: bool,
    kwargs_plot: Dict[str, Any] = None,
):
    # Use subplot to grid to center second row of subplots
    n_subplots = len(values)

    pivoted_df_list = []

    for value in values:
        df_tightness = df
        if tight_cases is not None:
            df_tightness = df[df["is_tight"] == tight_cases]
        df_val = df_tightness.pivot_table(
            index=xval,
            columns=yval,
            values=value,
            aggfunc=aggfunc,
        )
        df_val = df_val.sort_index(ascending=False)
        # df_val = df_val.sort_values(ascending=False, by=yval)
        pivoted_df_list.append(df_val)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    gs = GridSpec(nrows=2, ncols=4)

    ax0 = plt.subplot(gs[0, :2])
    ax1 = plt.subplot(gs[0, 2:])
    ax2 = plt.subplot(gs[1, 1:3])

    axs = [ax0, ax1, ax2]

    for lv_plot in range(n_subplots - 1):
        sns.heatmap(
            pivoted_df_list[lv_plot],
            ax=axs[lv_plot],
            cbar=False,
            **kwargs_plot,
        )

    sns.heatmap(
        pivoted_df_list[n_subplots - 1],
        ax=ax2,
        # cbar_kws={"label": label, "fraction": 0.046, "pad": 0.04},
        cbar_kws={"label": label, "fraction": 0.035, "pad": 0.04},
        # cbar=False,
        **kwargs_plot,
    )

    return fig, axs


def comparison_heatmaps(
    df: pd.DataFrame,
    xval: str,
    yval: str,
    values: List[str],
    aggfunc: str,
    label: str,
    tight_cases: bool,
    kwargs_plot: Dict[str, Any] = None,
) -> Tuple[Any, List[plt.Axes]]:

    n_subplots = len(values)
    pivoted_df_list = []
    for value in values:
        df_tightness = df
        if tight_cases is not None:
            df_tightness = df[df["is_tight"] == tight_cases]
        df_val = df_tightness.pivot_table(
            index=xval,
            columns=yval,
            values=value,
            aggfunc=aggfunc,
        )
        df_val = df_val.sort_index(ascending=False)
        # df_val = df_val.sort_values(ascending=False, by=yval)
        pivoted_df_list.append(df_val)
        bop = 1
    fig, axs = plt.subplots(1, n_subplots, sharey=False)
    if n_subplots == 1:
        axs = [axs]
    fig.set_figheight(4)
    fig.set_figwidth(4 * n_subplots)
    axs: plt.Axes = axs
    for lv_plot in range(n_subplots - 1):
        sns.heatmap(
            pivoted_df_list[lv_plot],
            ax=axs[lv_plot],
            cbar=False,
            **kwargs_plot,
        )
    if n_subplots > 1:

        sns.heatmap(
            pivoted_df_list[n_subplots - 1],
            ax=axs[n_subplots - 1],
            cbar_kws={"label": label},
            **kwargs_plot,
        )

    return fig, axs


def comparison_barcharts(
    df: pd.DataFrame,
    xval: str,
    values: List[str],
    aggfunc: str,
    hue_label: str,
    groupby: str,
    kwargs_plot: Dict[str, Any] = None,
) -> Tuple[Any, plt.Axes]:
    """
    Generates a grouped bar chart where hues are given by the 'values' list
    and the data is grouped by the provided x-axis value (xval).

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - xval (str): The column to group by for the x-axis.
    - values (List[str]): The list of values to plot (these will be used as hues).
    - aggfunc (str): The aggregation function (e.g., 'mean').
    - hue_label (str): Label for the hue.
    - groupby (str): The column to group data by on the x-axis.
    - kwargs_plot (Dict[str, Any]): Additional arguments for seaborn plot customization.

    Returns:
    - fig, ax (Tuple): The figure and axis objects for the plot.
    """

    # Reshape the dataframe to long format
    df_long = df.melt(
        id_vars=[xval, groupby],
        value_vars=values,
        var_name=hue_label,
        value_name="value",
    )

    # Aggregate data based on xval, groupby, and the values
    aggregated_data = df_long.groupby([xval, groupby, hue_label], as_index=False).agg(
        {"value": aggfunc}
    )

    # Create the barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=xval, y="value", hue=hue_label, data=aggregated_data, ax=ax, **kwargs_plot
    )

    # Adding labels and title
    ax.set_xlabel(xval)
    ax.set_ylabel(f"{aggfunc.capitalize()} of {hue_label}")
    ax.set_title(
        f"Grouped Bar Plot: {aggfunc.capitalize()} of {hue_label} grouped by {xval}"
    )

    # Adjust the legend and position if needed
    ax.legend(title=hue_label, loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    return fig, ax


class Analysis:
    def __init__(
        self,
        analysis_id: str,
        parameter_dict_list: List[Dict[str, Any]],
        param_settings_list_breakdown: List[Dict[str, Any]],
        root_dir: str,
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
        self.default_arg_dict = default_arg_dict

    def make_dirs(self):
        Path(self.top_level_fig_dir).mkdir(exist_ok=True, parents=True)
        Path(self.top_level_result_dir).mkdir(exist_ok=True, parents=True)
        Path(self.top_level_factor_graph_dir).mkdir(exist_ok=True, parents=True)

    def factor_graph_setup(self, args):
        logger.info(
            f"Setting up factor graphs for\
                    Analysis ID: {self.analysis_id}\n\
                    Root dir: {self.root_dir}\n\
                    Top level factor graph dir: {self.top_level_factor_graph_dir}\n"
        )
        ParameterDictList = self.parameter_dict_list
        top_level_factor_graph_dir = self.top_level_factor_graph_dir
        self.fg_fnames_list_of_lists = []
        # TODO: I think parameter_dict overrrides n_trials_per_run to None.
        for parameter_dict in tqdm(ParameterDictList):
            # fig_dir = os.path.join(top_level_fig_dir, noise_run_id)
            # Path(fig_dir).mkdir(exist_ok=True, parents=True)
            arg_dict = self.default_arg_dict.copy()
            arg_dict.update(parameter_dict)
            args = ParserArguments(**arg_dict)

            # bit of a dirty hack
            if args.experiment_type == "lost_in_the_woods":
                tmax_overall_liw = 1200
                allowed_time = tmax_overall_liw - args.tmin_overall_liw
                implied_num_subsequences = allowed_time / (
                    args.n_poses * (args.pose_spacing - 1)
                )
                num_subsequences = min(
                    [int(np.floor(implied_num_subsequences)), args.max_num_subsequences]
                )
                args.n_trials_per_run = num_subsequences
            run_id = string_id_from_parameters(parameter_dict)
            fg_dir = os.path.join(
                top_level_factor_graph_dir,
                run_id,
            )
            Path(fg_dir).mkdir(exist_ok=True, parents=True)

            logger.info(f"Factor graph names: {parameter_dict}")

            fg_fnames = [
                os.path.join(fg_dir, f"fg_{lv_seed}.pkl")
                for lv_seed in range(args.n_trials_per_run)
            ]
            self.factor_graph_setup_from_args(args, fnames=fg_fnames)
            self.fg_fnames_list_of_lists.append(fg_fnames)

    def factor_graph_setup_from_args(
        self, args, fnames: List[str], seed_override: int = None
    ):
        for lv_seed in range(len(fnames)):
            if seed_override is None:
                np.random.seed(lv_seed)
            else:
                np.random.seed(seed_override)
            fg_path = fnames[lv_seed]
            if args.problem_type == "se2":
                noise_parameters = gen_se2.Se2CaseNoiseParameters.from_args(args)
                fg = gen_se2.create_se2_factor_graph(
                    n_poses=args.n_poses,
                    n_landmarks=args.n_landmarks,
                    landmark_spread=args.landmark_spread,
                    meas_per_landmark=args.meas_per_landmark,
                    fraction_removal=args.fraction_removal,
                    uda_fraction=args.uda_fraction,
                    noise_parameters=noise_parameters,
                    add_noise=not args.noiseless,
                    prior_landmark_location=args.prior_landmark_location,
                )
                if args.subtask == "localization":
                    subtask_factors.convert_to_localization_task(fg)
                if args.subtask == "mapping":
                    subtask_factors.convert_to_mapping_task(fg)
            if args.problem_type == "toy":
                fg = test_cases.generate_random_toy_case(
                    nx=args.nx,
                    n_components_per_factor=args.n_components_per_factor,
                    scale_center=args.scale_center,
                    scale_offset=args.scale_offset,
                    scale_Q=args.scale_Q,
                )
            print(fg_path)
            with open(
                fg_path,
                "wb",
            ) as f:
                pickle.dump(fg, f)

    def progress_check(self):
        # For now this only works if n_trials_per_run is in parameter_dict
        logger.info(
            f"Progress check for for\
                    Analysis ID: {self.analysis_id}\n\
                    Root dir: {self.root_dir}\n\
                    Top level fig dir: {self.top_level_fig_dir}\n\
                    Top level factor graph dir: {self.top_level_factor_graph_dir}\n\
                    Top level result dir: {self.top_level_result_dir}"
        )
        df_dir = self.top_level_result_dir
        df_fname = os.path.join(df_dir, "results_df.pkl")
        ParameterDictList = self.parameter_dict_list
        top_level_result_dir = self.top_level_result_dir
        top_level_fig_dir = self.top_level_fig_dir
        top_level_factor_graph_dir = self.top_level_factor_graph_dir

        n_trials_per_run = ParameterDictList[0]["n_trials_per_run"]
        n_files = 0
        for p_dict in ParameterDictList:
            run_id = string_id_from_parameters(p_dict)
            results_dir = os.path.join(
                top_level_result_dir,
                run_id,
            )
            path = Path(results_dir)

            if path.exists() and path.is_dir():
                # Count the number of files (not directories) in the directory
                n_files += len(
                    [
                        f
                        for f in path.iterdir()
                        if f.is_file() and "mc_result_comparison" in f.name
                    ]
                )
        total_num_files = len(ParameterDictList) * n_trials_per_run
        print(f"Done: {n_files} of {total_num_files}")
        bop = 1
        # for lv_res in range(args.n_trials_per_run):
        #     with open(
        #         os.path.join(results_dir, f"mc_result_comparison_{lv_res}.pkl"),
        #         "rb",
        #     ) as f:
        #         try:
        #             mc_result_comparison: MonteCarloResultComparison = (
        #                 pickle.load(f)
        #             )
        #         except:
        #             logger.warning(f"Failed to load file {f}")
        #             continue

    def run_analysis(
        self,
        delete_existing_dataframe: bool,
        postprocess_only: bool,
        continue_trials: bool,
        overriden_factor_graph_fnames: List[str] = None,
        log_eigval_tightness_thresh: float = 5,
    ):
        logger.info(
            f"Running analysis for\
                    Analysis ID: {self.analysis_id}\n\
                    Root dir: {self.root_dir}\n\
                    Top level fig dir: {self.top_level_fig_dir}\n\
                    Top level factor graph dir: {self.top_level_factor_graph_dir}\n\
                    Top level result dir: {self.top_level_result_dir}"
        )
        df_dir = self.top_level_result_dir
        df_fname = os.path.join(df_dir, "results_df.pkl")
        DELETE_EXISTING_DATAFRAME = delete_existing_dataframe
        POSTPROCESS_ONLY = postprocess_only
        CONTINUE_TRIALS = continue_trials
        ParameterDictList = self.parameter_dict_list
        top_level_result_dir = self.top_level_result_dir
        top_level_fig_dir = self.top_level_fig_dir
        top_level_factor_graph_dir = self.top_level_factor_graph_dir

        LOG_EIGVAL_TIGHTNESS_THRESH = log_eigval_tightness_thresh
        if DELETE_EXISTING_DATAFRAME:
            Path(df_fname).unlink(missing_ok=True)

        try:
            with open(df_fname, "rb") as f:
                df = pickle.load(f)
            bop = 1
        except:
            result_dict = {
                "duality_gap": [],
                "log_eigval_ratio": [],
                "sdp_da_wrong_count": [],
                "mm_dr_init_da_wrong_count": [],
                "mm_gt_init_da_wrong_count": [],
                # "mm_sdp_init_da_wrong_count": [],
                "sdp_error_wrt_gt": [],
                "sdp_error_wrt_mm_gt_init_rot": [],
                "dr_error_wrt_mm_gt_init_rot": [],
                # "mm_sdp_init_error_wrt_mm_gt_init_rot": [],
                "sdp_error_wrt_mm_gt_init_pos": [],
                "dr_error_wrt_mm_gt_init_pos": [],
                # "mm_sdp_init_error_wrt_mm_gt_init_pos": [],
                "DeltaC_max_mix_dr_init": [],
                "DeltaC_sdp": [],
                "run_number": [],
                "n_pose": [],
                "n_landmarks": [],
                "fig_path": [],
                "gt_association_da_wrong_count": [],
                "gt_association_error_rot": [],
                "gt_association_error_pos": [],
                "cost_mm_gt": [],
                "cost_mm_dr": [],
                "cost_sdp": [],
                "cost_gt_association": [],
            }
            logger.info("Starting going through parameter dict list.. ")
            for parameter_dict in tqdm(ParameterDictList):
                for param_name in parameter_dict.keys():
                    if param_name not in result_dict:
                        result_dict[param_name] = []

                arg_dict = self.default_arg_dict
                arg_dict.update(parameter_dict)
                args = ParserArguments(**arg_dict)
                if args.experiment_type == "lost_in_the_woods":
                    # Figure out allowable number of subsequences.
                    tmax_overall_liw = 1200
                    allowed_time = tmax_overall_liw - args.tmin_overall_liw
                    implied_num_subsequences = allowed_time / (
                        args.n_poses * (args.pose_spacing - 1)
                    )
                    num_subsequences = min(
                        [
                            int(np.floor(implied_num_subsequences)),
                            args.max_num_subsequences,
                        ]
                    )
                    args.n_trials_per_run = num_subsequences
                if args.n_landmarks > 2 and args.n_poses > 2:
                    args.n_jobs = 2

                # fig_dir = os.path.join(top_level_fig_dir, noise_run_id)
                # Path(fig_dir).mkdir(exist_ok=True, parents=True)
                run_id = string_id_from_parameters(parameter_dict)
                results_dir = os.path.join(
                    top_level_result_dir,
                    run_id,
                )
                # Can we completely separate this out from running the mc? TODO one thing at a time.
                # factor graph directory and figure directory
                fg_dir = os.path.join(
                    top_level_factor_graph_dir,
                    run_id,
                )
                fig_dir = os.path.join(
                    top_level_fig_dir,
                    run_id,
                )
                Path(fg_dir).mkdir(exist_ok=True, parents=True)
                Path(fig_dir).mkdir(exist_ok=True, parents=True)

                logger.info(f"Factor graph names: {parameter_dict}")

                fg_fnames = [
                    os.path.join(fg_dir, f"fg_{lv_seed}.pkl")
                    for lv_seed in range(args.n_trials_per_run)
                ]

                if not POSTPROCESS_ONLY:
                    if overriden_factor_graph_fnames is None:
                        print(parameter_dict)
                        self.factor_graph_setup(args)
                    args.results_dir = os.path.join(top_level_result_dir)
                    args.run_id = string_id_from_parameters(parameter_dict)
                    logger.info(f"Running for parameter dict {parameter_dict}")

                    if overriden_factor_graph_fnames is None:
                        main_monte_carlo(args, fg_fnames, CONTINUE_TRIALS)
                    else:
                        main_monte_carlo(
                            args, overriden_factor_graph_fnames, CONTINUE_TRIALS
                        )

                for lv_res in range(args.n_trials_per_run):
                    with open(
                        os.path.join(results_dir, f"mc_result_comparison_{lv_res}.pkl"),
                        "rb",
                    ) as f:
                        try:
                            mc_result_comparison: MonteCarloResultComparison = (
                                pickle.load(f)
                            )
                        except:
                            logger.warning(f"Failed to load file {f}")
                            continue
                        mc_result_comparison.set_solutions_dict()
                        mc_result_comparison.set_data_associations_dict()
                        # Question: How to plot measurements?
                        # mc_result has the measurements in it...

                        plt.figure()
                        ax = plt.gca()
                        fig, ax = mc_result_comparison.plot_solutions(ax)
                        fig_path = os.path.join(
                            fig_dir, f"mc_result_comparison_{lv_res}.pdf"
                        )
                        plt.savefig(fig_path)
                        plt.close()

                        plt.figure()
                        ax = plt.gca()
                        fig, ax = mc_result_comparison.plot_solutions(
                            ax, plot_measurements=True, labels_to_exclude=[]
                        )
                        fig_path = os.path.join(
                            fig_dir,
                            f"mc_result_comparison_{lv_res}_with_measurements.pdf",
                        )
                        plt.savefig(fig_path)
                        plt.close()

                        summary = mc_result_comparison.summarize()
                        summary_file = os.path.join(
                            fig_dir, f"mc_result_comparison_{lv_res}_summary.txt"
                        )
                        with open(summary_file, "w") as f:
                            f.write(summary)
                        # Process costs between MM and SDP and GT...
                        mc_summary: MonteCarloSummary = (
                            mc_result_comparison.mc_result_sdp.postprocess()
                        )

                        default_result_names_of_interest = [
                            "duality_gap",
                            "log_eigval_ratio",
                            "n_pose",
                            "n_landmarks",
                            "sdp_error_wrt_gt",
                        ]

                        result_dict["duality_gap"].append(mc_summary.duality_gap)
                        result_dict["log_eigval_ratio"].append(
                            mc_summary.log_eigval_ratio
                        )
                        result_dict["n_pose"].append(args.n_poses)
                        result_dict["n_landmarks"].append(args.n_landmarks)

                        error_dict_rot, error_dict_pos = (
                            mc_result_comparison.compute_errors()
                        )
                        da_error_dict = (
                            mc_result_comparison.compute_data_association_errors()
                        )

                        result_dict["sdp_da_wrong_count"].append(da_error_dict["SDP"])

                        result_dict["mm_dr_init_da_wrong_count"].append(
                            da_error_dict["Max-Mix DR Init"]
                        )
                        result_dict["mm_gt_init_da_wrong_count"].append(
                            da_error_dict["Max-Mix GT Init"]
                        )
                        result_dict["gt_association_da_wrong_count"].append(
                            da_error_dict["GT Associations"]
                        )

                        # result_dict["mm_sdp_init_da_wrong_count"].append(
                        #     da_error_dict["Max-Mix SDP Init"]
                        # )

                        result_dict["sdp_error_wrt_gt"].append(mc_summary.state_error)
                        # ROTATION ERRORS
                        result_dict["sdp_error_wrt_mm_gt_init_rot"].append(
                            error_dict_rot["SDP"]
                        )
                        result_dict["dr_error_wrt_mm_gt_init_rot"].append(
                            error_dict_rot["Max-Mix DR Init"]
                        )
                        result_dict["gt_association_error_rot"].append(
                            error_dict_rot["GT Associations"]
                        )

                        # GT Associations

                        # result_dict["mm_sdp_init_error_wrt_mm_gt_init_rot"].append(
                        #     error_dict_rot["Max-Mix SDP Init"]
                        # )

                        # POSITION ERRORS
                        result_dict["sdp_error_wrt_mm_gt_init_pos"].append(
                            error_dict_pos["SDP"]
                        )
                        result_dict["dr_error_wrt_mm_gt_init_pos"].append(
                            error_dict_pos["Max-Mix DR Init"]
                        )
                        result_dict["gt_association_error_pos"].append(
                            error_dict_pos["GT Associations"]
                        )
                        result_dict["fig_path"].append(fig_path)

                        # result_dict["mm_sdp_init_error_wrt_mm_gt_init_pos"].append(
                        #     error_dict_pos["Max-Mix SDP Init"]
                        # )
                        # mm_sdp_init_da_wrong_count: [],
                        # mm_sdp_init_error_wrt_mm_gt_init_pos: [],
                        # mm_sdp_init_error_wrt_mm_gt_init_rot
                        for param_name, val in parameter_dict.items():
                            if param_name not in default_result_names_of_interest:
                                result_dict[param_name].append(val)
                        # for key, value in result_dict.items():
                        #     print(f"{key}: {len(value)}")
                        # "DeltaC_max_mix_dr_init": [],
                        # "DeltaC_sdp": [],
                        cost_mm_gt = mc_result_comparison.result_max_mix_gt_init[
                            "summary"
                        ].cost[-1]
                        cost_mm_dr = mc_result_comparison.result_max_mix_dr_init[
                            "summary"
                        ].cost[-1]
                        cost_gt_association = (
                            mc_result_comparison.result_gt_association["summary"].cost[
                                -1
                            ]
                        )
                        cost_sdp = mc_summary.est_cost / 2

                        result_dict["DeltaC_max_mix_dr_init"].append(
                            cost_mm_dr - cost_mm_gt
                        )
                        result_dict["DeltaC_sdp"].append(
                            mc_summary.est_cost / 2 - cost_mm_gt
                        )
                        result_dict["run_number"].append(lv_res)

                        result_dict["cost_mm_gt"].append(cost_mm_gt)
                        result_dict["cost_mm_dr"].append(cost_mm_dr)
                        result_dict["cost_gt_association"].append(cost_gt_association)
                        result_dict["cost_sdp"].append(cost_sdp)

            df = pd.DataFrame.from_dict(result_dict)
            df["is_tight"] = df["log_eigval_ratio"].apply(
                lambda x: x > LOG_EIGVAL_TIGHTNESS_THRESH
            )

            with open(df_fname, "wb") as f:
                pickle.dump(df, f)
        df["sdp_da_wrong"] = (df["sdp_da_wrong_count"] != 0).astype(int)
        df["mm_dr_init_da_wrong"] = (df["mm_dr_init_da_wrong_count"] != 0).astype(int)
        df["mm_gt_init_da_wrong"] = (df["mm_gt_init_da_wrong_count"] != 0).astype(int)
        self.df = df
        return df

    def describe_each_parameter_breakdown(self, columns=["log_eigval_ratio"]):
        param_settings_list_breakdown = self.param_settings_list_breakdown
        df = self.df
        df = self.df
        param_settings_names = [
            string_id_from_parameters(param_settings)
            for param_settings in param_settings_list_breakdown
        ]
        for param_settings, name in zip(
            param_settings_list_breakdown, param_settings_names
        ):
            df_filter = df.copy()
            for param_name, val in param_settings.items():
                df_filter = df_filter[df_filter[param_name] == val]
            df_filter = df_filter[columns]
            print(f"Dataframe description for {param_settings}")
            print(df_filter.describe())
        print(df.columns)

    def plots(
        self,
        xval="rel_noise_scale_cov",
        yval="rel_landmark_meas_cov",
        kwargs_heatmap={},
    ):
        param_settings_list_breakdown = self.param_settings_list_breakdown
        top_level_fig_dir = self.top_level_fig_dir
        df = self.df
        print(df.describe())
        param_settings_names = [
            string_id_from_parameters(param_settings)
            for param_settings in param_settings_list_breakdown
        ]
        for param_settings, name in zip(
            param_settings_list_breakdown, param_settings_names
        ):
            logger.info(f"Dataframe analysis for {param_settings}")

            fig_dir = os.path.join(top_level_fig_dir, name)
            Path(fig_dir).mkdir(exist_ok=True, parents=True)
            df_filter = df.copy()
            for param_name, val in param_settings.items():
                df_filter = df_filter[df_filter[param_name] == val]
            self.plot_df(
                df_filter, fig_dir, xval=xval, yval=yval, kwargs_heatmap=kwargs_heatmap
            )

    # Next TODO: Clean up sim plots.
    def plot_df(
        self,
        df: pd.DataFrame,
        fig_dir,
        xval="rel_landmark_meas_cov",
        yval="rel_landmark_meas_cov",
        kwargs_heatmap={},
    ):
        label_rel_pos_noise = "Rel. Pos Noise Scale"
        label_landmark_meas = "Landmark Pos. Cov."
        sns.set_style("whitegrid")
        palette = sns.color_palette("colorblind")

        params = {
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "text.usetex": True,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
        plt.rcParams.update(params)

        """-----------TIGHTNESS - MEDIAN OF EIGENVALUE RATIO-----------"""
        plt.figure()
        sns.boxplot(
            x="rel_noise_scale",
            y="log_eigval_ratio",
            hue="rel_landmark_meas_cov",
            data=df,
            palette=palette,
        )
        plt.axhline(y=6, color="red", linestyle="--", linewidth=2)
        plt.xlabel(label_rel_pos_noise)
        plt.ylabel("Log$_{10}$ Eigenvalue Ratio")
        # plt.legend(title="rel_landmark_meas_cov")
        plt.legend(
            title=label_landmark_meas,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=4,
        )
        plt.savefig(os.path.join(fig_dir, "log_eigval_ratio_boxplot.pdf"))

        df2 = df.pivot_table(
            index="rel_landmark_meas_cov",
            columns="rel_noise_scale",
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

        """-----------POSITION ERRORS-----------"""
        from pathlib import Path

        err_boxplot_fig_dir = os.path.join(fig_dir, "error_boxplots")
        Path(err_boxplot_fig_dir).mkdir(exist_ok=True, parents=True)
        params = {
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "text.usetex": True,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
        plt.rcParams.update(params)
        for tight, tight_label in zip([True, False], ["tight", "nontight"]):
            fig, axs = comparison_heatmaps(
                df,
                xval="rel_landmark_meas_cov",
                yval="rel_noise_scale",
                values=[
                    "sdp_error_wrt_mm_gt_init_pos",
                    "dr_error_wrt_mm_gt_init_pos",
                    # "mm_sdp_init_error_wrt_mm_gt_init_pos",
                ],
                aggfunc="median",
                tight_cases=tight,
                label="Median Position Error",
                kwargs_plot=dict(
                    {
                        "vmin": 0,
                        "vmax": 1.5,
                        "cmap": "coolwarm",
                        "annot_kws": {"fontsize": 10},
                        "fmt": ".1e",
                    },
                    **kwargs_heatmap,
                ),
            )
            # ax1: plt.Axes = ax1
            # ax2: plt.Axes = ax2
            # for ax in axs:
            #     ax.set_xlabel(label_rel_pos_noise, fontsize=15)
            #     ax.set_ylabel(label_landmark_meas, fontsize=15)
            # set_cmap_axes_parameters(axs, [label_rel_pos_noise, label_landmark_meas])
            for ax in axs:
                # ax.xaxis.set_tick_params(labelsize=14)
                # ax.yaxis.set_tick_params(labelsize=14)
                ax.set_xlabel(label_rel_pos_noise)
                ax.set_ylabel(label_landmark_meas)
                # ax.tick_params(axis="x", labelrotation=45)
                # ax.tick_params(axis="y", labelrotation=45)
                # ax.xaxis.label.set_size(15)
                # ax.yaxis.label.set_size(15)
            axs[0].set_title("SDP")
            axs[1].set_title("MM DR INIT")
            axs[1].set_ylabel(None)
            # axs[2].set_title("MM SDP INIT")
            plt.savefig(os.path.join(fig_dir, f"pos_errors_vs_noise_{tight_label}.pdf"))

        for tight, tight_label in zip([True, False], ["tight", "nontight"]):
            for value in [
                "sdp_error_wrt_mm_gt_init_pos",
                "dr_error_wrt_mm_gt_init_pos",
                "sdp_error_wrt_mm_gt_init_rot",
                "dr_error_wrt_mm_gt_init_rot",
            ]:
                plt.figure()
                sns.boxplot(
                    x="rel_noise_scale",
                    y=value,
                    hue="rel_landmark_meas_cov",
                    data=df[df["is_tight"] == tight],
                    palette=palette,
                )
                plt.xlabel("rel_noise_scale")
                plt.ylabel(value)
                plt.ylim(-0.05, 0.5)
                plt.legend(title="rel_landmark_meas_cov")
                plt.savefig(
                    os.path.join(err_boxplot_fig_dir, f"{value}_{tight_label}.pdf")
                )

        """-----------ROTATION ERRORS-----------"""
        for tight, tight_label in zip([True, False], ["tight", "nontight"]):

            fig, axs = comparison_heatmaps(
                df,
                xval="rel_landmark_meas_cov",
                yval="rel_noise_scale",
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
            for ax in axs:
                ax.set_xlabel(label_rel_pos_noise, fontsize=15)
                ax.set_ylabel(label_landmark_meas, fontsize=15)

            axs[0].set_title("SDP")
            axs[1].set_title("MM DR INIT")
            # axs[2].set_title("MM SDP INIT")
            plt.savefig(os.path.join(fig_dir, f"rot_errors_vs_noise_{tight_label}.pdf"))

        for tight, tightlabel in zip([True, False], ["tight", "nontight"]):
            fig, axs = comparison_heatmap_three_plots(
                df,
                xval="rel_landmark_meas_cov",
                yval="rel_noise_scale",
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
                        "vmax": 1,
                        "cmap": "coolwarm",
                        "annot_kws": {"fontsize": 10},
                        "fmt": ".2f",
                    },
                    **kwargs_heatmap,
                ),
            )
            # set_cmap_axes_parameters(axs, [label_rel_pos_noise, label_landmark_meas])
            for ax in axs:
                # ax.xaxis.set_tick_params(labelsize=14)
                # ax.yaxis.set_tick_params(labelsize=14)
                ax.set_xlabel(label_rel_pos_noise)
                ax.set_ylabel(label_landmark_meas)
                # ax.xaxis.label.set_size(15)
                # ax.yaxis.label.set_size(15)
            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            ax1.set_aspect("equal")
            ax2.set_aspect("equal")
            ax2.set_ylabel(None)
            # ax2.set_yticklabels([])
            ax3.set_aspect("equal")
            ax1.set_title("SDP")
            ax2.set_title("MM DR INIT")
            ax3.set_title("MM GT INIT")
            for ax in axs:
                ax.tick_params(axis="x", labelrotation=45)
                ax.tick_params(axis="y", labelrotation=45)
            plt.savefig(os.path.join(fig_dir, f"data_associations_{tightlabel}.pdf"))

            # Next breakdown. For cases where data associations are NOT the same...
            # Plot percentage of cases where SDP cost < Max Mix GT Cost.
            df["sdp_minus_gt_association_cost"] = (
                df["cost_sdp"] - df["cost_gt_association"]
            )
            df1 = df[df["sdp_da_wrong_count"] != df["gt_association_da_wrong_count"]]

            df_val = df1[df1["is_tight"] == tight].pivot_table(
                index="rel_landmark_meas_cov",
                columns="rel_noise_scale",
                values="sdp_minus_gt_association_cost",
                aggfunc="max",
            )
            df_val = df_val.sort_index(ascending=False)
            if df_val.empty:
                print(f"Empty dataframe for tight {tight}")
                continue
            plt.figure()
            sns.heatmap(
                df_val,
                annot=True,
                cbar=False,
                cmap="coolwarm",
            )
            plt.savefig(
                os.path.join(
                    fig_dir, f"sdp_minus_gt_association_cost_{tightlabel}_max.pdf"
                )
            )

            df_val = df1[df1["is_tight"] == tight].pivot_table(
                index="rel_landmark_meas_cov",
                columns="rel_noise_scale",
                values="sdp_minus_gt_association_cost",
                aggfunc="mean",
            )
            df_val = df_val.sort_index(ascending=False)
            plt.figure()
            sns.heatmap(
                df_val,
                annot=True,
                cbar=False,
                cmap="coolwarm",
            )
            plt.savefig(
                os.path.join(
                    fig_dir, f"sdp_minus_gt_association_cost_{tightlabel}_mean.pdf"
                )
            )

        df_val = df.pivot_table(
            index="rel_landmark_meas_cov",
            columns="rel_noise_scale",
            values="is_tight",
            aggfunc="mean",
        )
        df_val = df_val.sort_index(ascending=False)
        plt.figure()
        sns.heatmap(
            df_val,
            annot=True,
            cbar=True,
            cmap=sns.cubehelix_palette(as_cmap=True),
        )
        plt.xlabel(label_rel_pos_noise)
        plt.ylabel(label_landmark_meas)
        plt.savefig(os.path.join(fig_dir, f"tightness_fraction.pdf"))

        df_long = df_val.reset_index().melt(
            id_vars="rel_landmark_meas_cov",
            var_name="rel_noise_scale",
            value_name="mean_is_tight",
        )
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x="rel_noise_scale",
            y="mean_is_tight",
            hue="rel_landmark_meas_cov",
            data=df_long,
            palette=palette,
        )
        # Labels and title
        plt.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

        plt.xlabel("Relative pose measurement noise multiplier")
        plt.ylabel("Fraction of tight cases")
        plt.legend(
            title="Relative landmark position noise standard deviation",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=5,
        )

        # Save the plot
        plt.savefig(os.path.join(fig_dir, "tightness_fraction_barchart.pdf"))

    def checks(self):
        # 1. Data associations... MM GT init matching the relaxation.
        #    - Check a) Costs and b) Get those cases to see their summaries and plots.
        # Have to filter dataframe.
        df: pd.DataFrame = self.df
        print(df.describe())
        df_subset = df[df["sdp_da_wrong"] != df["mm_gt_init_da_wrong"]]
        df_subset = df_subset[df["is_tight"] == True]
        df_subset = df_subset[df["n_poses"] == 3]

        print(df_subset.describe())
        print(df_subset)
        print(df_subset.columns)
        pd.options.display.max_colwidth = 1000
        print(df_subset[["sdp_da_wrong", "mm_gt_init_da_wrong", "DeltaC_sdp"]])
        # TODO: Add heatmap making sure cost attained by SDP is lower than GT init.


def get_analyses() -> Dict[str, Analysis]:
    return {
        "analysis_debug": Analysis(
            analysis_id="analysis_debug",
            parameter_dict_list=[
                {
                    "solver": "mosek",
                    "rel_landmark_meas_cov": landmark_noise,
                    # "rel_rot_kappa_langevin_inverse": rot_noise,
                    # "rel_pos_cov": rel_pos_noise,
                    "rel_noise_scale": rel_noise_scale,
                    "noiseless": noiseless,
                    "n_trials_per_run": 1,
                    "n_poses": n_pose,
                    "n_landmarks": n_landmarks,
                }
                for landmark_noise, rel_noise_scale, noiseless, n_pose, n_landmarks in itertools.product(
                    [1e-2, 0.5],
                    [1, 20],
                    [False],
                    [2, 3],
                    [1, 2],
                )
            ],
            param_settings_list_breakdown=[
                # {
                # "noiseless": True,
                # },
                {"noiseless": False},
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict(),
        ),
        "analysis_testing": Analysis(
            analysis_id="analysis_testing",
            parameter_dict_list=[
                {
                    "solver": "mosek",
                    "rel_landmark_meas_cov": landmark_noise,
                    # "rel_rot_kappa_langevin_inverse": rot_noise,
                    # "rel_pos_cov": rel_pos_noise,
                    "rel_noise_scale": rel_noise_scale,
                    "noiseless": noiseless,
                    "n_trials_per_run": 3,
                    "n_poses": n_pose,
                    "n_landmarks": n_landmarks,
                }
                for landmark_noise, rel_noise_scale, noiseless, n_pose, n_landmarks in itertools.product(
                    [1e-2, 0.1],
                    [1, 5],
                    # [1e-2, 0.1, 0.5, 1, 1.5, 2, 3],
                    # [1, 5, 10, 20, 40, 60, 80, 100],
                    [False],
                    [3, 4],
                    [2, 3],
                )
            ],
            param_settings_list_breakdown=[
                # {
                #     "noiseless": True,
                # },
                {"noiseless": False},
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict(),
        ),
        "analysis_coarse": Analysis(
            analysis_id="analysis_coarse",
            parameter_dict_list=[
                {
                    "solver": "mosek",
                    "rel_landmark_meas_cov": landmark_noise,
                    # "rel_rot_kappa_langevin_inverse": rot_noise,
                    # "rel_pos_cov": rel_pos_noise,
                    "rel_noise_scale": rel_noise_scale,
                    "noiseless": noiseless,
                    "n_trials_per_run": 4,
                    "n_poses": n_pose,
                    "n_landmarks": n_landmarks,
                    "rel_rot_kappa_langevin_inverse_base_val": 0.1**2,
                    "rel_pos_base_val": 0.866**2,
                }
                for landmark_noise, rel_noise_scale, noiseless, n_pose, n_landmarks in itertools.product(
                    [1, 3, 4],
                    [0.1, 1, 30, 40],
                    [False],
                    [5],
                    [3],
                )
            ],
            param_settings_list_breakdown=[
                # {
                #     "noiseless": True,
                # },
                {"noiseless": False},
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict(),
        ),
        "analysis_fine": Analysis(
            analysis_id="analysis_fine",
            parameter_dict_list=[
                {
                    "solver": "mosek",
                    "rel_landmark_meas_cov": landmark_noise,
                    # "rel_rot_kappa_langevin_inverse": rot_noise,
                    # "rel_pos_cov": rel_pos_noise,
                    "rel_noise_scale": rel_noise_scale,
                    "noiseless": noiseless,
                    "n_trials_per_run": 10,
                    "n_poses": n_pose,
                    "n_landmarks": n_landmarks,
                    "rel_rot_kappa_langevin_inverse_base_val": 0.1**2,
                    "rel_pos_base_val": 0.866**2,
                }
                for landmark_noise, rel_noise_scale, noiseless, n_pose, n_landmarks in itertools.product(
                    [0.5, 1, 2, 3, 4, 5],
                    [0.1, 1, 10, 20, 30, 40, 50, 60],
                    [False],
                    [3, 5],
                    [2, 3],
                )
            ],
            param_settings_list_breakdown=[
                # {
                #     "noiseless": True,
                # },
                {"noiseless": False},
            ],
            root_dir=path_config.project_root_dir,
            default_arg_dict=default_arg_dict(),
        ),
    }
