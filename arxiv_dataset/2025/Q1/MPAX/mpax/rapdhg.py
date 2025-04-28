import abc
import logging
import timeit
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCSR, BCOO

from mpax.loop_utils import while_loop
from mpax.preprocess import rescale_problem
from mpax.restart import (
    run_restart_scheme,
    run_restart_scheme_feasibility_polishing,
    select_initial_primal_weight,
    unscaled_saddle_point_output,
)
from mpax.solver_log import (
    display_iteration_stats_heading,
    pdhg_final_log,
    setup_logger,
)
from mpax.termination import (
    cached_quadratic_program_info,
    check_termination_criteria,
    check_primal_feasibility,
    check_dual_feasibility,
)
from mpax.utils import (
    PdhgSolverState,
    QuadraticProgrammingProblem,
    RestartInfo,
    RestartParameters,
    RestartScheme,
    RestartToCurrentMetric,
    SaddlePointOutput,
    TerminationCriteria,
    TerminationStatus,
    ScaledQpProblem,
    ConvergenceInformation,
)
from mpax.feasibility_polishing import (
    init_primal_feasibility_polishing,
    set_dual_solution_to_zero,
    init_dual_feasibility_polishing,
    set_primal_solution_to_zero,
)
from mpax.iteration_stats_utils import compute_convergence_information

logger = logging.getLogger(__name__)


def estimate_maximum_singular_value(
    matrix: BCSR,
    probability_of_failure: float = 0.01,
    desired_relative_error: float = 0.1,
    seed: int = 1,
) -> tuple:
    """
    Estimate the maximum singular value of a sparse matrix using the power method.

    Parameters
    ----------
    matrix : BCSR
        The sparse matrix in BCSR format.
    probability_of_failure : float, optional
        The acceptable probability of failure.
    desired_relative_error : float, optional
        The desired relative error for the estimation.
    seed : int, optional
        The random seed for reproducibility.

    Returns
    -------
    tuple
        The estimated maximum singular value and the number of power iterations.
    """
    epsilon = 1.0 - (1.0 - desired_relative_error) ** 2
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (matrix.shape[1],))
    if isinstance(matrix, BCSR):
        matrix_transpose = BCSR.from_bcoo(matrix.to_bcoo().T)
    elif isinstance(matrix, BCOO):
        matrix_transpose = BCSR.from_bcoo(matrix.T)
    elif isinstance(matrix, jnp.ndarray):
        matrix_transpose = matrix.T
    number_of_power_iterations = 0

    def cond_fun(state):
        # Corresponds to the power_method_failure_probability in CuPDLP.jl
        x, number_of_power_iterations = state
        power_method_failure_probability = jax.lax.cond(
            # We have to use bitwise operators | instead of or in JAX.
            # Ref: https://github.com/jax-ml/jax/issues/3761#issuecomment-658456938
            (number_of_power_iterations < 2) | (epsilon <= 0.0),
            lambda _: 1.0,
            lambda _: (
                jax.lax.min(
                    0.824, 0.354 / jnp.sqrt(epsilon * (number_of_power_iterations - 1))
                )
                * jnp.sqrt(matrix.shape[1])
                * (1.0 - epsilon) ** (number_of_power_iterations - 0.5)
            ),
            operand=None,
        )
        return power_method_failure_probability > probability_of_failure

    def body_fun(state):
        x, number_of_power_iterations = state
        x = x / jnp.linalg.norm(x, 2)
        x = matrix_transpose @ (matrix @ x)
        return x, number_of_power_iterations + 1

    # while_loop() compiles cond_fun and body_fun, so while it can be combined with jit(), it’s usually unnecessary.
    x, number_of_power_iterations = while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(x, 0),
        maxiter=1000,
        unroll=False,
        jit=True,
    )
    return (
        jnp.sqrt(
            jnp.dot(x, matrix_transpose @ (matrix @ x)) / jnp.linalg.norm(x, 2) ** 2
        ),
        number_of_power_iterations,
    )


def compute_next_solution(
    problem: QuadraticProgrammingProblem,
    solver_state: PdhgSolverState,
    step_size: float,
    extrapolation_coefficient: float,
):
    """Compute the next primal and dual solutions.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    step_size : float
        The step size used in the PDHG algorithm.
    extrapolation_coefficient : float
        The extrapolation coefficient.

    Returns
    -------
    tuple
        The delta primal, delta primal product, and delta dual.
    """
    # Compute the next primal solution.
    # For LPs, momentum is not activated since both avg_primal_obj_product and current_primal_obj_product are zero vectors.
    momentum_coef = 1 / (1.0 + solver_state.solutions_count / 2.0)
    next_primal_solution = solver_state.current_primal_solution - (
        step_size / solver_state.primal_weight
    ) * (
        problem.objective_vector
        - solver_state.current_dual_product
        + (1 - momentum_coef) * solver_state.avg_primal_obj_product
        + momentum_coef * solver_state.current_primal_obj_product
    )
    # Projection.
    next_primal_solution = jnp.minimum(
        problem.variable_upper_bound,
        jnp.maximum(problem.variable_lower_bound, next_primal_solution),
    )
    delta_primal = next_primal_solution - solver_state.current_primal_solution
    delta_primal_product = problem.constraint_matrix @ delta_primal

    # Compute the next dual solution.
    next_dual_solution = solver_state.current_dual_solution + (
        solver_state.primal_weight * step_size
    ) * (
        problem.right_hand_side
        - (1 + extrapolation_coefficient) * delta_primal_product
        - solver_state.current_primal_product
    )
    next_dual_solution = jnp.where(
        problem.inequalities_mask,
        jnp.maximum(next_dual_solution, 0.0),
        next_dual_solution,
    )
    delta_dual_solution = next_dual_solution - solver_state.current_dual_solution
    return delta_primal, delta_primal_product, delta_dual_solution


def line_search(
    problem, solver_state, reduction_exponent, growth_exponent, step_size_limit_coef
):
    """Perform a line search to find a good step size.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    reduction_exponent : float
        The reduction exponent for adaptive step size.
    growth_exponent : float
        The growth exponent for adaptive step size.
    step_size_limit_coef: float
        The step size limit coefficient for adaptive step size.

    Returns
    -------
    tuple
        The delta_primal, delta_dual, delta_primal_product, step_size, and line_search_iter.
    """

    def cond_fun(line_search_state):
        step_size_limit = line_search_state[4]
        old_step_size = line_search_state[6]
        return jax.lax.cond(
            old_step_size <= step_size_limit,
            lambda _: False,
            lambda _: True,
            operand=None,
        )

    def body_fun(line_search_state):
        line_search_iter = line_search_state[0]
        step_size_limit = line_search_state[4]
        step_size = line_search_state[5]
        line_search_iter += 1
        delta_primal, delta_primal_product, delta_dual = compute_next_solution(
            problem, solver_state, step_size, 1.0
        )
        interaction = jnp.abs(jnp.dot(delta_primal_product, delta_dual))
        movement = 0.5 * solver_state.primal_weight * jnp.sum(
            jnp.square(delta_primal)
        ) + (0.5 / solver_state.primal_weight) * jnp.sum(jnp.square(delta_dual))

        step_size_limit = jax.lax.cond(
            interaction > 0,
            lambda _: movement / interaction * step_size_limit_coef,
            lambda _: jnp.inf,
            operand=None,
        )
        old_step_size = step_size
        first_term = (
            1
            - 1
            / (solver_state.num_steps_tried + line_search_iter + 1)
            ** reduction_exponent
        ) * step_size_limit
        second_term = (
            1
            + 1
            / (solver_state.num_steps_tried + line_search_iter + 1) ** growth_exponent
        ) * step_size
        step_size = jnp.minimum(first_term, second_term)

        return (
            line_search_iter,
            delta_primal,
            delta_dual,
            delta_primal_product,
            step_size_limit,
            step_size,
            old_step_size,
        )

    (
        line_search_iter,
        delta_primal,
        delta_dual,
        delta_primal_product,
        step_size_limit,
        step_size,
        old_step_size,
    ) = while_loop(
        cond_fun,
        body_fun,
        init_val=(
            0,
            jnp.zeros_like(solver_state.current_primal_solution),
            jnp.zeros_like(solver_state.current_dual_solution),
            jnp.zeros_like(solver_state.current_dual_solution),
            -jnp.inf,
            solver_state.step_size,
            solver_state.step_size,
        ),
        maxiter=10,
        unroll=False,
        jit=True,
    )

    return delta_primal, delta_dual, delta_primal_product, step_size, line_search_iter


@dataclass(eq=False)
class raPDHG(abc.ABC):
    """
    The raPDHG solver class.
    """

    verbose: bool = False
    debug: bool = False
    display_frequency: int = 10
    jit: bool = True
    unroll: bool = False
    termination_evaluation_frequency: int = 64
    optimality_norm: float = jnp.inf
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    eps_primal_infeasible: float = 1e-8
    eps_dual_infeasible: float = 1e-8
    # time_sec_limit: float = float("inf")
    iteration_limit: int = jnp.iinfo(jnp.int32).max
    l_inf_ruiz_iterations: int = 10
    l2_norm_rescaling: bool = False
    pock_chambolle_alpha: float = 1.0
    primal_importance: float = 1.0
    scale_invariant_initial_primal_weight: bool = True
    restart_scheme: int = RestartScheme.ADAPTIVE_KKT
    restart_to_current_metric: int = RestartToCurrentMetric.KKT_GREEDY
    restart_frequency_if_fixed: int = 1000
    artificial_restart_threshold: float = 0.36
    sufficient_reduction_for_restart: float = 0.2
    necessary_reduction_for_restart: float = 0.8
    primal_weight_update_smoothing: float = 0.5
    adaptive_step_size: bool = True
    adaptive_step_size_reduction_exponent: float = 0.3
    adaptive_step_size_growth_exponent: float = 0.6
    adaptive_step_size_limit_coef: float = 1.0
    warm_start: bool = False
    feasibility_polishing: bool = False
    eps_feas_polish: float = 1e-06
    infeasibility_detection: bool = True

    def check_config(self, is_lp: bool):
        if not is_lp:
            self.infeasibility_detection = False
            self.primal_weight_update_smoothing = 0.2
            self.adaptive_step_size = False

        self._termination_criteria = TerminationCriteria(
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
            eps_primal_infeasible=self.eps_primal_infeasible,
            eps_dual_infeasible=self.eps_dual_infeasible,
            # time_sec_limit=self.time_sec_limit,
            iteration_limit=self.iteration_limit,
        )
        self._restart_params = RestartParameters(
            restart_scheme=self.restart_scheme,
            restart_to_current_metric=self.restart_to_current_metric,
            restart_frequency_if_fixed=self.restart_frequency_if_fixed,
            artificial_restart_threshold=self.artificial_restart_threshold,
            sufficient_reduction_for_restart=self.sufficient_reduction_for_restart,
            necessary_reduction_for_restart=self.necessary_reduction_for_restart,
            primal_weight_update_smoothing=self.primal_weight_update_smoothing,
        )
        self._polishing_termination_criteria = TerminationCriteria(
            eps_abs=self.eps_feas_polish,
            eps_rel=self.eps_feas_polish,
            eps_primal_infeasible=self.eps_primal_infeasible,
            eps_dual_infeasible=self.eps_dual_infeasible,
            iteration_limit=self.iteration_limit,
        )

    def calculate_constant_step_size(
        self, primal_weight, iteration, last_step_size
    ) -> float:
        """Calculate the constant step size for the raPDHG algorithm.

        Parameters
        ----------
        primal_weight : float
            The primal weight.
        iteration : int
            The inner iteration counter, which will be set to zero if restart.
        last_step_size : float
            Step size in the last iteration.

        Returns
        -------
        float
            The constant step size.
        """
        next_step_size = (
            0.99
            * (2 + iteration)
            / (
                self._norm_Q / primal_weight
                + jnp.sqrt(
                    (2 + iteration) ** 2 * self._norm_A**2
                    + self._norm_Q**2 / primal_weight**2
                )
            )
        )
        # We use jnp.true_divide here since it returns inf for division by zero.
        step_size_limit = (1 + jnp.true_divide(1, iteration)) * last_step_size
        return jnp.minimum(next_step_size, step_size_limit)

    def initialize_solver_status(
        self,
        scaled_problem: ScaledQpProblem,
        initial_primal_solution: jnp.array,
        initial_dual_solution: jnp.array,
        is_lp: bool = True,
    ) -> PdhgSolverState:
        """Initialize the solver status for PDHG.

        Parameters
        ----------
        scaled_problem : ScaledQpProblem
            Scaled quadratic programming problem instance.
        initial_primal_solution : jnp.array
            The initial primal solution.
        initial_dual_solution : jnp.array
            The initial dual solution.

        Returns
        -------
        PdhgSolverState
            The initial solver status.
        """
        scaled_qp = scaled_problem.scaled_qp
        primal_size = len(scaled_qp.variable_lower_bound)
        dual_size = len(scaled_qp.right_hand_side)

        # Primal weight initialization
        if self.scale_invariant_initial_primal_weight:
            self._initial_primal_weight = select_initial_primal_weight(
                scaled_qp, 1.0, 1.0, self.primal_importance
            )
        else:
            self._initial_primal_weight = self.primal_importance

        # Step size computation
        if self.adaptive_step_size:
            if isinstance(scaled_qp.constraint_matrix, (BCOO, BCSR)):
                step_size = 1.0 / jnp.max(jnp.abs(scaled_qp.constraint_matrix.data))
            elif isinstance(scaled_qp.constraint_matrix, jnp.ndarray):
                step_size = 1.0 / jnp.max(jnp.abs(scaled_qp.constraint_matrix))
            else:
                raise ValueError("Unsupported matrix type.")
        else:
            # desired_relative_error = 0.2
            # maximum_singular_value, number_of_power_iterations = (
            #     estimate_maximum_singular_value(
            #         scaled_qp.constraint_matrix,
            #         probability_of_failure=0.001,
            #         desired_relative_error=desired_relative_error,
            #     )
            # )
            # step_size = (1 - desired_relative_error) / maximum_singular_value
            self._norm_A = estimate_maximum_singular_value(scaled_qp.constraint_matrix)[
                0
            ]
            self._norm_Q = jax.lax.cond(
                is_lp,
                lambda: 0.0,
                lambda: estimate_maximum_singular_value(scaled_qp.objective_matrix)[0],
            )
            step_size = 1.0  # Placeholder for step size.

        if self.warm_start:
            scaled_initial_primal_solution = (
                initial_primal_solution * scaled_problem.variable_rescaling
            )
            scaled_initial_dual_solution = (
                initial_dual_solution * scaled_problem.constraint_rescaling
            )
            scaled_initial_primal_product = (
                scaled_qp.constraint_matrix @ scaled_initial_primal_solution
            )
            scaled_initial_dual_product = (
                scaled_qp.constraint_matrix_t @ scaled_initial_dual_solution
            )
            scaled_primal_obj_product = (
                scaled_qp.objective_matrix @ scaled_initial_primal_solution
            )
        else:
            scaled_initial_primal_solution = jnp.zeros(primal_size)
            scaled_initial_dual_solution = jnp.zeros(dual_size)
            scaled_initial_primal_product = jnp.zeros(dual_size)
            scaled_initial_dual_product = jnp.zeros(primal_size)
            scaled_primal_obj_product = jnp.zeros(primal_size)
        solver_state = PdhgSolverState(
            current_primal_solution=scaled_initial_primal_solution,
            current_dual_solution=scaled_initial_dual_solution,
            current_primal_product=scaled_initial_primal_product,
            current_dual_product=scaled_initial_dual_product,
            current_primal_obj_product=scaled_primal_obj_product,
            solutions_count=0,
            weights_sum=0.0,
            step_size=step_size,
            primal_weight=self._initial_primal_weight,
            numerical_error=False,
            # total_number_iterations=0,
            avg_primal_solution=scaled_initial_primal_solution,
            avg_dual_solution=scaled_initial_dual_solution,
            avg_primal_product=scaled_initial_primal_product,
            avg_dual_product=scaled_initial_dual_product,
            avg_primal_obj_product=scaled_primal_obj_product,
            initial_primal_solution=scaled_initial_primal_solution,
            initial_dual_solution=scaled_initial_dual_solution,
            initial_primal_product=scaled_initial_primal_product,
            initial_dual_product=scaled_initial_dual_product,
            num_steps_tried=0,
            num_iterations=0,
            termination_status=TerminationStatus.UNSPECIFIED,
            delta_primal=jnp.zeros(primal_size),
            delta_dual=jnp.zeros(dual_size),
            delta_primal_product=jnp.zeros(dual_size),
        )

        last_restart_info = RestartInfo(
            primal_solution=scaled_initial_primal_solution,
            dual_solution=scaled_initial_dual_solution,
            primal_diff=jnp.zeros(primal_size),
            dual_diff=jnp.zeros(dual_size),
            primal_diff_product=jnp.zeros(dual_size),
            primal_product=scaled_initial_primal_product,
            dual_product=scaled_initial_dual_product,
            primal_obj_product=scaled_primal_obj_product,
        )
        return solver_state, last_restart_info

    def take_step(
        self, solver_state: PdhgSolverState, problem: QuadraticProgrammingProblem
    ) -> PdhgSolverState:
        """
        Take a PDHG step with adaptive step size.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        problem : QuadraticProgrammingProblem
            The problem being solved.
        """

        if self.adaptive_step_size:
            (
                delta_primal,
                delta_dual,
                delta_primal_product,
                step_size,
                line_search_iter,
            ) = line_search(
                problem,
                solver_state,
                self.adaptive_step_size_reduction_exponent,
                self.adaptive_step_size_growth_exponent,
                self.adaptive_step_size_limit_coef,
            )
        else:
            extrapolation_coefficient = solver_state.solutions_count / (
                solver_state.solutions_count + 1.0
            )
            step_size = jax.lax.cond(
                problem.is_lp,
                lambda: solver_state.step_size,
                lambda: self.calculate_constant_step_size(
                    solver_state.primal_weight,
                    solver_state.solutions_count,
                    solver_state.step_size,
                ),
            )
            delta_primal, delta_primal_product, delta_dual = compute_next_solution(
                problem, solver_state, step_size, extrapolation_coefficient
            )
            line_search_iter = 1

        next_primal_solution = solver_state.current_primal_solution + delta_primal
        next_primal_product = solver_state.current_primal_product + delta_primal_product
        next_primal_obj_product = problem.objective_matrix @ next_primal_solution
        next_dual_solution = solver_state.current_dual_solution + delta_dual
        next_dual_product = problem.constraint_matrix_t @ next_dual_solution

        ratio = step_size / (solver_state.weights_sum + step_size)
        next_avg_primal_solution = solver_state.avg_primal_solution + ratio * (
            next_primal_solution - solver_state.avg_primal_solution
        )
        next_avg_dual_solution = solver_state.avg_dual_solution + ratio * (
            next_dual_solution - solver_state.avg_dual_solution
        )
        next_avg_primal_product = solver_state.avg_primal_product + ratio * (
            next_primal_product - solver_state.avg_primal_product
        )
        next_avg_dual_product = solver_state.avg_dual_product + ratio * (
            next_dual_product - solver_state.avg_dual_product
        )
        next_avg_primal_obj_product = solver_state.avg_primal_obj_product + ratio * (
            next_primal_obj_product - solver_state.avg_primal_obj_product
        )
        new_solutions_count = solver_state.solutions_count + 1
        new_weights_sum = solver_state.weights_sum + step_size

        return PdhgSolverState(
            current_primal_solution=next_primal_solution,
            current_dual_solution=next_dual_solution,
            current_primal_product=next_primal_product,
            current_dual_product=next_dual_product,
            current_primal_obj_product=next_primal_obj_product,
            avg_primal_solution=next_avg_primal_solution,
            avg_dual_solution=next_avg_dual_solution,
            avg_primal_product=next_avg_primal_product,
            avg_dual_product=next_avg_dual_product,
            avg_primal_obj_product=next_avg_primal_obj_product,
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            delta_primal=delta_primal,
            delta_dual=delta_dual,
            delta_primal_product=delta_primal_product,
            solutions_count=new_solutions_count,
            weights_sum=new_weights_sum,
            step_size=step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried + line_search_iter,
            num_iterations=solver_state.num_iterations + 1,
            termination_status=TerminationStatus.UNSPECIFIED,
        )

    def take_multiple_steps(
        self, solver_state: PdhgSolverState, problem: QuadraticProgrammingProblem
    ) -> PdhgSolverState:
        """
        Take multiple PDHG step with adaptive step size.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        problem : QuadraticProgrammingProblem
            The problem being solved.
        """
        new_solver_state = jax.lax.fori_loop(
            lower=0,
            upper=self.termination_evaluation_frequency,
            body_fun=lambda i, x: self.take_step(x, problem),
            init_val=solver_state,
        )
        return new_solver_state

    def initial_iteration_update(
        self,
        solver_state,
        last_restart_info,
        should_terminate,
        scaled_problem,
        qp_cache,
    ):
        """The inner loop of PDLP algorithm.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        last_restart_info : RestartInfo
            The information of the last restart.
        should_terminate : bool
            Whether the algorithm should terminate.
        scaled_problem : ScaledQpProblem
            The scaled quadratic programming problem.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        tuple
            The updated solver state, the updated last restart info, whether to terminate, the scaled problem, and the cached quadratic programming information.
        """
        # Skip termination check for initial iterations
        restarted_solver_state, new_last_restart_info = run_restart_scheme(
            scaled_problem.scaled_qp,
            solver_state,
            last_restart_info,
            self._restart_params,
            self.optimality_norm,
        )

        new_solver_state = self.take_step(
            restarted_solver_state, scaled_problem.scaled_qp
        )
        new_solver_state.termination_status = TerminationStatus.UNSPECIFIED
        return (
            new_solver_state,
            new_last_restart_info,
            False,
            scaled_problem,
            qp_cache,
        )

    def main_iteration_update(
        self,
        solver_state,
        last_restart_info,
        should_terminate,
        scaled_problem,
        qp_cache,
        ci,
    ):
        """The inner loop of PDLP algorithm.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        last_restart_info : RestartInfo
            The information of the last restart.
        should_terminate : bool
            Whether the algorithm should terminate.
        scaled_problem : QuadraticProgrammingProblem
            The scaled quadratic programming problem.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        tuple
            The updated solver state, the updated last restart info, whether to terminate, the scaled problem, and the cached quadratic programming information.
        """
        # Check for termination
        new_should_terminate, new_termination_status, new_convergence_information = (
            check_termination_criteria(
                scaled_problem,
                solver_state,
                self._termination_criteria,
                qp_cache,
                solver_state.numerical_error,
                1.0,
                self.optimality_norm,
                average=True,
                infeasibility_detection=self.infeasibility_detection,
            )
        )

        restarted_solver_state, new_last_restart_info = run_restart_scheme(
            scaled_problem.scaled_qp,
            solver_state,
            last_restart_info,
            self._restart_params,
            self.optimality_norm,
        )

        new_solver_state = self.take_multiple_steps(
            restarted_solver_state, scaled_problem.scaled_qp
        )
        new_solver_state.termination_status = new_termination_status
        return (
            new_solver_state,
            new_last_restart_info,
            new_should_terminate,
            scaled_problem,
            qp_cache,
            new_convergence_information,
        )

    def primal_feasibility_polishing(self, solver_state, scaled_problem, qp_cache):
        """Perform primal feasibility polishing.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        scaled_problem : ScaledQpProblem
            The original problem and scaled problem data.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        jnp.array
            The primal solution.
        bool
            Whether primal feasibility succeeds.
        """
        (
            primal_feasibility_problem,
            primal_feasibility_solver_state,
            last_restart_info,
        ) = init_primal_feasibility_polishing(
            scaled_problem, solver_state, self._initial_primal_weight
        )
        (new_solver_state, last_restart_info, should_terminate, _, _) = while_loop(
            cond_fun=lambda state: state[2] == False,
            body_fun=lambda state: self.primal_feasibility_polishing_iterate(*state),
            init_val=(
                primal_feasibility_solver_state,
                last_restart_info,
                False,
                primal_feasibility_problem,
                qp_cache,
            ),
            maxiter=self.iteration_limit,
            unroll=self.unroll,
            jit=self.jit,
        )
        return new_solver_state.avg_primal_solution, should_terminate

    def primal_feasibility_polishing_iterate(
        self,
        primal_polishing_solver_state,
        last_restart_info,
        should_terminate,
        primal_feasibility_problem,
        qp_cache,
    ):
        zeroed_dual_solver_state = set_dual_solution_to_zero(
            primal_polishing_solver_state
        )
        restarted_primal_polishing_solver_state, new_last_restart_info = (
            run_restart_scheme_feasibility_polishing(
                primal_feasibility_problem.scaled_qp,
                primal_polishing_solver_state,
                zeroed_dual_solver_state,
                last_restart_info,
                self._restart_params,
            )
        )

        new_primal_polishing_solver_state = self.take_multiple_steps(
            restarted_primal_polishing_solver_state,
            primal_feasibility_problem.scaled_qp,
        )
        new_should_terminate = check_primal_feasibility(
            primal_feasibility_problem,
            new_primal_polishing_solver_state,
            self._polishing_termination_criteria,
            qp_cache,
            1.0,
            self.termination_evaluation_frequency * self.display_frequency,
            self.optimality_norm,
        )
        return (
            new_primal_polishing_solver_state,
            new_last_restart_info,
            new_should_terminate,
            primal_feasibility_problem,
            qp_cache,
        )

    def dual_feasibility_polishing(self, solver_state, scaled_problem, qp_cache):
        """Perform dual feasibility polishing.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        scaled_problem : ScaledQpProblem
            The original problem and scaled problem data.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        jnp.array
            The dual solution.
        bool
            Whether dual feasibility succeeds.
        """
        dual_feasibility_problem, dual_feasibility_solver_state, last_restart_info = (
            init_dual_feasibility_polishing(
                scaled_problem, solver_state, self._initial_primal_weight
            )
        )

        (new_solver_state, last_restart_info, should_terminate, _, _) = while_loop(
            cond_fun=lambda state: state[2] == False,
            body_fun=lambda state: self.dual_feasibility_polishing_iterate(*state),
            init_val=(
                dual_feasibility_solver_state,
                last_restart_info,
                False,
                dual_feasibility_problem,
                qp_cache,
            ),
            maxiter=self.iteration_limit,
            unroll=self.unroll,
            jit=self.jit,
        )
        return new_solver_state.avg_dual_solution, should_terminate

    def dual_feasibility_polishing_iterate(
        self,
        dual_polishing_solver_state,
        last_restart_info,
        should_terminate,
        dual_feasibility_problem,
        qp_cache,
    ):
        zeroed_primal_solver_state = set_primal_solution_to_zero(
            dual_polishing_solver_state
        )
        restarted_dual_polishing_solver_state, new_last_restart_info = (
            run_restart_scheme_feasibility_polishing(
                dual_feasibility_problem.scaled_qp,
                zeroed_primal_solver_state,
                dual_polishing_solver_state,
                last_restart_info,
                self._restart_params,
            )
        )

        new_dual_polishing_solver_state = self.take_multiple_steps(
            restarted_dual_polishing_solver_state, dual_feasibility_problem.scaled_qp
        )

        new_should_terminate = check_dual_feasibility(
            dual_feasibility_problem,
            new_dual_polishing_solver_state,
            self._polishing_termination_criteria,
            qp_cache,
            1.0,
            self.termination_evaluation_frequency * self.display_frequency,
            self.optimality_norm,
        )

        return (
            new_dual_polishing_solver_state,
            new_last_restart_info,
            new_should_terminate,
            dual_feasibility_problem,
            qp_cache,
        )

    def optimize(
        self,
        original_problem: QuadraticProgrammingProblem,
        initial_primal_solution=None,
        initial_dual_solution=None,
    ) -> SaddlePointOutput:
        """
        Main algorithm: given parameters and LP problem, return solutions.

        Parameters
        ----------
        original_problem : QuadraticProgrammingProblem
            The quadratic programming problem to be solved.
        initial_primal_solution : jnp.array, optional
            The initial primal solution.
        initial_dual_solution : jnp.array, optional
            The initial dual solution.

        Returns
        -------
        SaddlePointOutput
            The solution to the optimization problem.
        """
        setup_logger(self.verbose, self.debug)
        # validate(original_problem)
        # config_check(params)
        self.check_config(original_problem.is_lp)
        qp_cache = cached_quadratic_program_info(
            original_problem, norm_ord=self.optimality_norm
        )

        precondition_start_time = timeit.default_timer()
        scaled_problem = rescale_problem(
            self.l_inf_ruiz_iterations,
            self.l2_norm_rescaling,
            self.pock_chambolle_alpha,
            original_problem,
        )
        precondition_time = timeit.default_timer() - precondition_start_time
        logger.info("Preconditioning Time (seconds): %.2e", precondition_time)

        solver_state, last_restart_info = self.initialize_solver_status(
            scaled_problem,
            initial_primal_solution,
            initial_dual_solution,
            original_problem.is_lp,
        )

        # Iteration loop
        display_iteration_stats_heading()

        iteration_start_time = timeit.default_timer()
        # Initial iterations, where restart will be checked every iteration.
        (solver_state, last_restart_info, should_terminate, _, _) = while_loop(
            cond_fun=lambda state: state[2] == False,
            body_fun=lambda state: self.initial_iteration_update(*state),
            init_val=(solver_state, last_restart_info, False, scaled_problem, qp_cache),
            maxiter=10,
            unroll=self.unroll,
            jit=self.jit,
        )

        (solver_state, last_restart_info, should_terminate, _, _, ci) = while_loop(
            cond_fun=lambda state: state[2] == False,
            body_fun=lambda state: self.main_iteration_update(*state),
            init_val=(
                solver_state,
                last_restart_info,
                False,
                scaled_problem,
                qp_cache,
                ConvergenceInformation(),
            ),
            maxiter=self.iteration_limit,
            unroll=self.unroll,
            jit=self.jit,
        )
        iteration_time = timeit.default_timer() - iteration_start_time

        if self.feasibility_polishing:
            feasibility_polishing_start_time = timeit.default_timer()
            polished_primal_solution, primal_feasibility = (
                self.primal_feasibility_polishing(
                    solver_state, scaled_problem, qp_cache
                )
            )
            polished_dual_solution, dual_feasibility = self.dual_feasibility_polishing(
                solver_state, scaled_problem, qp_cache
            )
            feasibility_polishing_time = (
                timeit.default_timer() - feasibility_polishing_start_time
            )
            (
                solver_state.avg_primal_solution,
                solver_state.avg_primal_product,
                solver_state.avg_dual_solution,
                solver_state.avg_dual_product,
                solver_state.avg_primal_obj_product,
            ) = jax.lax.cond(
                primal_feasibility & dual_feasibility,
                lambda: (
                    polished_primal_solution,
                    scaled_problem.scaled_qp.constraint_matrix
                    @ polished_primal_solution,
                    polished_dual_solution,
                    scaled_problem.scaled_qp.constraint_matrix_t
                    @ polished_dual_solution,
                    scaled_problem.scaled_qp.objective_matrix
                    @ polished_primal_solution,
                ),
                lambda: (
                    solver_state.avg_primal_solution,
                    solver_state.avg_primal_product,
                    solver_state.avg_dual_solution,
                    solver_state.avg_dual_product,
                    solver_state.avg_primal_obj_product,
                ),
            )
            ci = compute_convergence_information(
                scaled_problem.original_qp,
                qp_cache,
                solver_state.avg_primal_solution / scaled_problem.variable_rescaling,
                solver_state.avg_dual_solution / scaled_problem.constraint_rescaling,
                self.abs_rel / self.rel_eps,
                solver_state.avg_primal_product * scaled_problem.constraint_rescaling,
                solver_state.avg_dual_product * scaled_problem.variable_rescaling,
                solver_state.avg_primal_obj_product * scaled_problem.variable_rescaling,
                self.optimality_norm,
            )
        else:
            feasibility_polishing_time = 0

        timing = {
            "Preconditioning": precondition_time,
            "Iteration loop": iteration_time,
            "Feasibility polishing": feasibility_polishing_time,
        }

        # Log the stats of the final iteration.
        pdhg_final_log(
            scaled_problem.scaled_qp,
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.num_iterations,
            solver_state.termination_status,
            timing,
            ci,
        )
        return unscaled_saddle_point_output(
            scaled_problem,
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.termination_status,
            solver_state.num_iterations - 1,
            ci,
            timing,
        )
