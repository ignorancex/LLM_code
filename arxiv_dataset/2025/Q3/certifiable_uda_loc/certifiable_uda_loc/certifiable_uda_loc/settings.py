from dataclasses import dataclass
from typing import List
import attrs
from attrs import define, field


@define
class ProblemSolutionSettings:
    sparse_bool_cont_variables: bool = field()
    create_constraints_from_nullspace: bool = field()
    create_discrete_variable_constraints_from_nullspace: bool = field()
    use_sparse_matrices: bool = field()
    sparsify_interfactor_discrete_constraints: bool = field()
    sparsify_A_th_premultiplication: bool = field()
    sparsify_cx_premultiplication_of_A_th: bool = field()
    sparsify_cxi_cxj_premultiplication_of_A_th: bool = field()
    sparsify_analytic_moment_constraints: bool = field()
    sparsify_analytic_moment_constraints_2: bool = field()
    sparsify_analytic_moment_constraints_3: bool = field()
    sparsify_off_diag_boolean: bool = field()
    sparsify_bool_product_moment: bool = field()
    locked_first_pose: bool = field()
    solver_primal: bool = field()
    solver_cost_matrix_adjust: bool = field()
    prior_landmark_location: str = field(default="random")
    solver: str = field(default="mosek")
    constraints_to_remove: List[str] = field(default=[])
    discrete_variable_constraints: List[str] = field(
        default=[
            "bool",
            "prod_ci_cj",
            "c_c_squared",
            "sum_one",
        ]
    )


@dataclass
class Se2CaseNoiseParameters:
    prior_rot_kappa_langevin_inverse: float
    prior_pos_cov: float
    prior_landmark_cov: float
    prior_landmark_noise_corrupt_cov: float
    rel_rot_kappa_langevin_inverse: float
    rel_pos_cov: float
    rel_landmark_meas_cov: float

    @staticmethod
    def from_args(args):
        rel_rot_kappa_langevin_inverse = (
            args.rel_rot_kappa_langevin_inverse_base_val * args.rel_noise_scale
        )
        rel_pos_cov = args.rel_pos_base_val * args.rel_noise_scale
        return Se2CaseNoiseParameters(
            args.prior_rot_kappa_langevin_inverse,
            args.prior_pos_cov,
            args.prior_landmark_cov,
            args.prior_landmark_noise_corrupt_cov,
            rel_rot_kappa_langevin_inverse,
            rel_pos_cov,
            args.rel_landmark_meas_cov,
        )


def problem_settings_from_args(args):
    problem_settings = ProblemSolutionSettings(
        sparse_bool_cont_variables=True,
        create_constraints_from_nullspace=args.create_constraints_from_nullspace,
        create_discrete_variable_constraints_from_nullspace=args.create_discrete_variable_constraints_from_nullspace,
        discrete_variable_constraints=args.discrete_variable_constraints,
        use_sparse_matrices=args.use_sparse_matrices_properly,
        sparsify_interfactor_discrete_constraints=not args.no_sparsify_interfactor_discrete_constraints,
        sparsify_A_th_premultiplication=not args.no_sparsify_A_th_premultiplication,
        sparsify_cx_premultiplication_of_A_th=not args.no_sparsify_cx_premultiplication_of_A_th,
        sparsify_cxi_cxj_premultiplication_of_A_th=not args.no_sparsify_cxi_cxj_premultiplication_of_A_th,
        sparsify_analytic_moment_constraints=not args.no_sparsify_analytic_moment_constraints,
        sparsify_analytic_moment_constraints_2=not args.no_sparsify_analytic_moment_constraints_2,
        sparsify_analytic_moment_constraints_3=not args.no_sparsify_analytic_moment_constraints_3,
        sparsify_off_diag_boolean=not args.no_sparsify_off_diag_boolean,
        # sparsify_off_diag_boolean=False,
        sparsify_bool_product_moment=not args.no_sparsify_bool_product_moment,
        locked_first_pose=args.locked_first_pose,
        solver_primal=args.solver_primal,
        solver_cost_matrix_adjust=args.solver_cost_matrix_adjust,
        prior_landmark_location=args.prior_landmark_location,
        solver=args.solver,
    )
    return problem_settings
