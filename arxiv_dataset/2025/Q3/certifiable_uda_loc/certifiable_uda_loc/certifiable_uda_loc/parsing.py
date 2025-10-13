import argparse
import ast
import sys
from dataclasses import dataclass, field
from typing import List
import certifiable_uda_loc.path_configuration as path_config

# Example usage
# python 3_monte_carlo.py se2 --se2_args


# parser.add_argument(
#     "--load_from_disk", action="store_true", help="Flag to load data from disk"
# )
# parser.add_argument(
#     "--load_postprocessed_df_from_disk",
#     action="store_true",
#     help="Flag to load postprocessed DataFrame from disk",
# )
@dataclass
class ParserArguments:
    run_id: str = "test"
    verbose: bool = False
    results_dir: str = path_config.project_root_dir
    n_trials_per_run: int = 10
    create_constraints_from_nullspace: bool = False
    create_discrete_variable_constraints_from_nullspace: bool = False
    discrete_variable_constraints: List[str] = field(
        default_factory=lambda: ["bool", "prod_ci_cj", "sum_one"]
    )
    n_jobs: int = 10
    use_sparse_matrices_properly: bool = True

    # Sparsity arguments
    sparse_bool_cont_variables: bool = True
    no_sparsify_interfactor_discrete_constraints: bool = False
    no_sparsify_A_th_premultiplication: bool = False
    no_sparsify_cx_premultiplication_of_A_th: bool = False
    no_sparsify_cxi_cxj_premultiplication_of_A_th: bool = False
    no_sparsify_analytic_moment_constraints: bool = False
    no_sparsify_analytic_moment_constraints_2: bool = False
    no_sparsify_analytic_moment_constraints_3: bool = False
    no_sparsify_off_diag_boolean: bool = True
    no_sparsify_bool_product_moment: bool = False

    problem_type: str = "se2"  # Default problem type
    experiment_type: str = "simulation"  # "simulation" or "lost_in_the_woods"
    overall_time_bounds: List[float] = field(
        default_factory=lambda: [
            200,
            210,
        ]
    )

    max_num_subsequences: int = 1  # if using lost in the woods
    subtask: str = "localization"
    prior_rot_kappa_langevin_inverse: float = 0.01
    prior_pos_cov: float = 0.01
    # prior_landmark_cov: float = 0.01
    prior_landmark_cov: float = 200
    prior_landmark_noise_corrupt_cov: float = None

    # rel_rot_kappa_langevin_inverse: float = 0.01
    # rel_pos_cov: float = 0.01
    rel_rot_kappa_langevin_inverse_base_val: float = 0.01
    rel_pos_base_val: float = 0.01
    rel_noise_scale: float = 1
    rel_landmark_meas_cov: float = 0.01
    n_poses: int = 5
    n_landmarks: int = 2
    landmark_spread: float = 5
    meas_per_landmark: int = 1
    fraction_removal: float = 0
    uda_fraction: float = 1
    noiseless: bool = False
    locked_first_pose: bool = False
    solver: str = "mosek"
    solver_primal: bool = False
    solver_cost_matrix_adjust: bool = True
    prior_landmark_location: str = None  # "random" or "gt"

    # Only applicable for lost in the woods
    tmin_overall_liw: float = None
    num_subsequences: int = None
    # The number of poses in each subsection is specified using n_poses
    pose_spacing: float = None  # Amount of time between each pose
    landmarks_to_use: List[int] = None  # Landmark ids to use for lost in the woods
    use_ground_truth_for_relative_pose_liw: bool = True
    use_ground_truth_for_relative_landmark_meas_liw: bool = True
    unknown_data_association_liw: bool = True


def set_default_subparser(self, name, args=None, positional_args=0):
    # https://stackoverflow.com/questions/6365601/default-sub-command-or-handling-no-sub-command-with-argparse
    """default subparser selection. Call after setup, just before parse_args()
    name: is the name of the subparser to call by default
    args: if set is the argument list handed to parse_args()

    , tested with 2.7, 3.2, 3.3, 3.4
    it works with 2.6 assuming argparse is installed
    """
    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ["-h", "--help"]:  # global help if no subparser
            break
    else:
        for x in self._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            # insert default in last position before global positional
            # arguments, this implies no global options are specified after
            # first positional argument
            if args is None:
                sys.argv.insert(len(sys.argv) - positional_args, name)
            else:
                args.insert(len(args) - positional_args, name)


argparse.ArgumentParser.set_default_subparser = set_default_subparser


def get_parser() -> argparse.ArgumentParser:
    # Reference for subparsers
    # https://gist.github.com/amarao/36327a6f77b86b90c2bca72ba03c9d3a
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id", type=str, default="test", help="Unique identifier for the run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbosity",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=path_config.top_level_result_dir,
        help="Path to save results",
    )

    parser.add_argument(
        "--n_trials_per_run", type=int, default=1, help="Number of trials per run"
    )

    parser.add_argument(
        "--create_constraints_from_nullspace",
        action="store_true",
        default=False,
        help="Whether to create constraints from nullspace",
    )

    parser.add_argument(
        "--create_discrete_variable_constraints_from_nullspace",
        action="store_true",
        default=False,
        help="Whether to create discrete constraints from nullspace",
    )

    parser.add_argument(
        "--discrete_variable_constraints",
        nargs="+",
        type=str,
        default=["bool", "prod_ci_cj", "c_c_squared", "sum_one"],
        help="Set of constraints to use for discrete variables",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run",
    )

    parser.add_argument(
        "--use_sparse_matrices_properly",
        action="store_true",
        default=False,
        help="There was previously a mildly cursed way of initializing and using PolyMatrix. Then it was changed to be better. This flag allows us to change\
        between them to make sure we didnt break anything. ",
    )

    parser.add_argument(
        "--solver_primal",
        action="store_true",
        default=False,
        help="Use primal form for problem input to solver",
    )

    parser.add_argument(
        "--solver-cost-matrix-adjust",
        action="store_true",
        default=False,
        help="Adjust cost matrix for better problem conditioning",
    )

    sparsity_arguments = [
        "no_sparsify_interfactor_discrete_constraints",
        "no_sparsify_A_th_premultiplication",
        "no_sparsify_cx_premultiplication_of_A_th",
        "no_sparsify_cxi_cxj_premultiplication_of_A_th",
        "no_sparsify_analytic_moment_constraints",
        "no_sparsify_analytic_moment_constraints_2",
        "no_sparsify_analytic_moment_constraints_3",
        "no_sparsify_off_diag_boolean",
        "no_sparsify_bool_product_moment",
    ]

    # Loop over each argument and add it to the parser
    for arg in sparsity_arguments:
        parser.add_argument(
            f"--{arg}",
            action="store_true",
            default=False,
        )

    subparsers = parser.add_subparsers(dest="problem_type")
    se2_parser = subparsers.add_parser("se2", help="SE2 Parser")

    se2_parser.add_argument(
        "--subtask",
        type=str,
        help="One of [slam, localization, mapping]",
        default="localization",
    )

    se2_parser.add_argument(
        "--sparse_bool_cont_variables",
        action="store_true",
        default=False,
        help="Flag to add noise",
    )

    se2_parser.add_argument(
        "--prior_rot_kappa_langevin_inverse",
        type=float,
        default=0.01,
        help="Prior for rotation kappa Langevin inverse",
    )
    se2_parser.add_argument(
        "--prior_pos_cov",
        type=float,
        default=0.01,
        help="Prior for position covariance",
    )
    se2_parser.add_argument(
        "--locked_first_pose",
        action="store_true",
        default=False,
        help="Lock first pose to ground truth value instead of adding prior. ",
    )
    se2_parser.add_argument(
        "--prior_landmark_cov",
        type=float,
        default=0.01,
        help="Prior for landmark covariance",
    )
    se2_parser.add_argument(
        "--prior_landmark_noise_corrupt_cov",
        type=float,
        default=0.01,
        help="Prior for landmark noise corrupt covariance",
    )
    se2_parser.add_argument(
        "--rel_rot_kappa_langevin_inverse",
        type=float,
        default=0.01,
        help="Relative rotation kappa Langevin inverse",
    )
    se2_parser.add_argument(
        "--rel_pos_cov", type=float, default=0.01, help="Relative position covariance"
    )
    se2_parser.add_argument(
        "--rel_landmark_meas_cov",
        type=float,
        default=0.01,
        help="Relative landmark measurement covariance",
    )
    se2_parser.add_argument("--n_poses", type=int, default=2, help="Number of poses")

    se2_parser.add_argument(
        "--n_landmarks", type=int, default=2, help="Number of landmarks"
    )
    se2_parser.add_argument(
        "--landmark_spread", type=float, default=5, help="Spread of landmarks"
    )
    se2_parser.add_argument(
        "--meas_per_landmark",
        type=int,
        default=1,
        help="Number of measurements per landmark",
    )
    se2_parser.add_argument(
        "--fraction_removal",
        type=float,
        default=0,
        help="Fraction of measurements to remove",
    )
    se2_parser.add_argument(
        "--uda_fraction", type=float, default=1, help="Fraction of UDA"
    )
    se2_parser.add_argument(
        "--noiseless",
        action="store_true",
        default=False,
        help="Flag to add noise",
    )
    se2_parser.add_argument(
        "--prior_landmark_location",
        default="random",
        help="There needs to be a prior on landmark locations to remove unobservability issues (swapping\
        landmark IDS gives same solution). Can be set to random or gt",
    )

    # TODO: Toy Example subparser.
    toy_parser = subparsers.add_parser("toy", help="Toy Example Parser")

    toy_parser.add_argument(
        "--nx",
        type=int,
        default=2,
    )
    toy_parser.add_argument(
        "--n_components_per_factor", type=int, nargs="+", default=[2, 2]
    )
    toy_parser.add_argument("--scale_center", type=float, default=5)
    toy_parser.add_argument("--scale_offset", type=float, default=10)
    toy_parser.add_argument("--scale_Q", type=float, default=3)

    parser.set_default_subparser("se2")

    return parser


# def add_general_arguments_to_parser(parser: argparse.ArgumentParser):

#     parser.add_argument(
#         "--results_path",
#         type=str,
#         default=path_config.top_level_result_dir,
#         help="Path to save results",
#     )
#     parser.add_argument(
#         "--load_from_disk", action="store_true", help="Flag to load data from disk"
#     )
#     parser.add_argument(
#         "--load_postprocessed_df_from_disk",
#         action="store_true",
#         help="Flag to load postprocessed DataFrame from disk",
#     )

#     return parser


# # def parse_n_components_per_factor_list(input_string: str):
# #     list_of_lists = [
# #         ast.literal_eval(inner.strip()) for inner in input_string.split(",")
# #     ]
# #     return list_of_lists


# def parse_n_components_per_factor_list(input_str):
#     # Remove spaces and split the input string into parts
#     input_str = input_str.replace(" ", "")  # Remove spaces
#     # Wrap in brackets and replace single brackets with a list of lists format
#     input_str = f"[{input_str}]"
#     return ast.literal_eval(input_str)
