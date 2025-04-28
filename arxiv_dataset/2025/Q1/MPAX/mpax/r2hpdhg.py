import logging
import timeit
from dataclasses import dataclass

import jax.numpy as jnp
from jax.lax import cond

from mpax.loop_utils import while_loop
from mpax.preprocess import rescale_problem
from mpax.rapdhg import compute_next_solution, line_search, raPDHG
from mpax.restart import (
    compute_new_primal_weight,
    restart_criteria_met_fixed_point,
    unscaled_saddle_point_output,
    weighted_norm,
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
    RestartScheme,
    RestartToCurrentMetric,
    SaddlePointOutput,
    TerminationStatus,
    ConvergenceInformation,
)
from mpax.feasibility_polishing import (
    set_dual_solution_to_zero,
    set_primal_solution_to_zero,
    init_primal_feasibility_polishing,
    init_dual_feasibility_polishing,
)
from mpax.iteration_stats_utils import compute_convergence_information

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class r2HPDHG(raPDHG):
    """
    The r2HPDHG solver class.
    """

    verbose: bool = False
    debug: bool = False
    display_frequency: int = 10
    jit: bool = True
    unroll: bool = False
    termination_evaluation_frequency: int = 100
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
    artificial_restart_threshold: float = 0.2
    sufficient_reduction_for_restart: float = 0.2
    necessary_reduction_for_restart: float = 0.6
    primal_weight_update_smoothing: float = 0.6
    adaptive_step_size: bool = True
    adaptive_step_size_reduction_exponent: float = 0.4
    adaptive_step_size_growth_exponent: float = 0.8
    adaptive_step_size_limit_coef: float = 0.2
    warm_start: bool = False
    feasibility_polishing: bool = False
    eps_feas_polish: float = 1e-06
    infeasibility_detection: bool = True

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
            delta_primal, delta_primal_product, delta_dual = compute_next_solution(
                problem, solver_state, solver_state.step_size, 1.0
            )
            step_size = solver_state.step_size
            line_search_iter = 1

        # Compute the weight according to the stepsize.
        new_solutions_count = solver_state.solutions_count + 1
        new_weights_sum = solver_state.weights_sum + solver_state.step_size
        initial_step_size = cond(
            solver_state.initial_step_size == 0,
            lambda _: step_size,
            lambda _: solver_state.initial_step_size,
            operand=None,
        )
        weight = (new_weights_sum) / (new_weights_sum + initial_step_size)
        next_primal_solution = (
            weight * (solver_state.current_primal_solution + 2 * delta_primal)
            + (1 - weight) * solver_state.initial_primal_solution
        )
        next_primal_product = (
            weight * (solver_state.current_primal_product + 2 * delta_primal_product)
            + (1 - weight) * solver_state.initial_primal_product
        )
        next_dual_solution = (
            weight * (solver_state.current_dual_solution + 2 * delta_dual)
            + (1 - weight) * solver_state.initial_dual_solution
        )
        next_dual_product = problem.constraint_matrix_t @ next_dual_solution

        return PdhgSolverState(
            current_primal_solution=next_primal_solution,
            current_dual_solution=next_dual_solution,
            current_primal_product=next_primal_product,
            current_dual_product=next_dual_product,
            current_primal_obj_product=jnp.zeros_like(next_primal_solution),
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            avg_primal_solution=jnp.zeros_like(next_primal_solution),
            avg_dual_solution=jnp.zeros_like(next_dual_solution),
            avg_primal_product=jnp.zeros_like(next_primal_product),
            avg_dual_product=jnp.zeros_like(next_dual_product),
            avg_primal_obj_product=jnp.zeros_like(next_primal_solution),
            solutions_count=new_solutions_count,
            weights_sum=new_weights_sum,
            step_size=step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried + line_search_iter,
            num_iterations=solver_state.num_iterations + 1,
            termination_status=TerminationStatus.UNSPECIFIED,
            delta_primal=delta_primal,
            delta_dual=delta_dual,
            delta_primal_product=delta_primal_product,
            initial_step_size=initial_step_size,
        )

    def perform_restart(
        self, solver_state, last_restart_info, kkt_reduction_ratio, problem
    ):
        # Take a pure PDHG step to get the new solution and set it as the initial solution for the outer iteration.
        # Use the pure PDHG step solution, instead of the Halpen PDHG step solution, as the initial solution for the restart.
        # Solver state has been updated to Halpen PDHG step solution, therefore, we need to retrieve the pure PDHG step solution.
        restart_length = solver_state.solutions_count
        # weight = 1 / solver_state.solutions_count
        weight = solver_state.initial_step_size / solver_state.weights_sum
        # Retrieve the last iteration solution and product.
        last_iteration_primal_solution = (
            (1 + weight) * solver_state.current_primal_solution
            - weight * solver_state.initial_primal_solution
            - solver_state.delta_primal
        )
        last_iteration_dual_solution = (
            (1 + weight) * solver_state.current_dual_solution
            - weight * solver_state.initial_dual_solution
            - solver_state.delta_dual
        )
        last_iteration_primal_product = (
            (1 + weight) * solver_state.current_primal_product
            - weight * solver_state.initial_primal_product
            - solver_state.delta_primal_product
        )
        last_iteration_dual_product = (
            problem.constraint_matrix_t @ solver_state.current_dual_solution
        )
        last_iteration_solver_state = PdhgSolverState(
            current_primal_solution=last_iteration_primal_solution,
            current_dual_solution=last_iteration_dual_solution,
            current_primal_product=last_iteration_primal_product,
            current_dual_product=last_iteration_dual_product,
            current_primal_obj_product=jnp.zeros_like(last_iteration_primal_solution),
            avg_primal_solution=solver_state.avg_primal_solution,
            avg_dual_solution=solver_state.avg_dual_solution,
            avg_primal_product=solver_state.avg_primal_product,
            avg_dual_product=solver_state.avg_dual_product,
            avg_primal_obj_product=jnp.zeros_like(solver_state.avg_primal_solution),
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            solutions_count=solver_state.solutions_count,
            weights_sum=solver_state.weights_sum,
            step_size=solver_state.step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried,
            num_iterations=solver_state.num_iterations - 1,
            termination_status=TerminationStatus.UNSPECIFIED,
        )
        restarted_solver_state = self.take_step(last_iteration_solver_state, problem)
        restarted_solver_state.initial_step_size = restarted_solver_state.step_size
        restarted_solver_state.initial_primal_solution = last_iteration_primal_solution
        restarted_solver_state.initial_dual_solution = last_iteration_dual_solution
        restarted_solver_state.initial_primal_product = last_iteration_primal_product
        restarted_solver_state.initial_dual_product = last_iteration_dual_product

        primal_norm_params = (
            1 / restarted_solver_state.step_size * restarted_solver_state.primal_weight
        )
        dual_norm_params = (
            1 / restarted_solver_state.step_size / restarted_solver_state.primal_weight
        )
        primal_distance_moved_last_restart_period = weighted_norm(
            restarted_solver_state.initial_primal_solution
            - last_restart_info.primal_solution,
            primal_norm_params,
        ) / jnp.sqrt(solver_state.primal_weight)
        dual_distance_moved_last_restart_period = weighted_norm(
            restarted_solver_state.initial_dual_solution
            - last_restart_info.dual_solution,
            dual_norm_params,
        ) * jnp.sqrt(solver_state.primal_weight)
        new_last_restart_info = RestartInfo(
            primal_solution=restarted_solver_state.initial_primal_solution,
            dual_solution=restarted_solver_state.initial_dual_solution,
            primal_diff=restarted_solver_state.delta_primal,
            dual_diff=restarted_solver_state.delta_dual,
            primal_diff_product=restarted_solver_state.delta_primal_product,
            primal_product=restarted_solver_state.initial_primal_product,
            dual_product=restarted_solver_state.initial_dual_product,
            primal_obj_product=restarted_solver_state.current_primal_obj_product,
            last_restart_length=restart_length,
            primal_distance_moved_last_restart_period=primal_distance_moved_last_restart_period,
            dual_distance_moved_last_restart_period=dual_distance_moved_last_restart_period,
            reduction_ratio_last_trial=kkt_reduction_ratio,
        )

        restarted_solver_state.primal_weight = compute_new_primal_weight(
            new_last_restart_info,
            solver_state.primal_weight,
            self.primal_weight_update_smoothing,
        )
        restarted_solver_state.solutions_count = 0
        restarted_solver_state.weights_sum = 0.0

        return restarted_solver_state, new_last_restart_info

    def run_restart_scheme(
        self,
        problem: QuadraticProgrammingProblem,
        solver_state: PdhgSolverState,
        last_restart_info: RestartInfo,
    ):
        """
        Check restart criteria based on current and average KKT residuals.

        Parameters
        ----------
        problem : QuadraticProgrammingProblem
            The quadratic programming problem instance.
        solver_state : PdhgSolverState
            The current solver state.
        last_restart_info : RestartInfo
            Information from the last restart.

        Returns
        -------
        tuple
            The new solver state, and the new last restart info.
        """
        do_restart, kkt_reduction_ratio = cond(
            solver_state.solutions_count == 0,
            lambda: (False, last_restart_info.reduction_ratio_last_trial),
            lambda: restart_criteria_met_fixed_point(
                self._restart_params, solver_state, last_restart_info
            ),
        )
        return cond(
            do_restart,
            lambda: self.perform_restart(
                solver_state, last_restart_info, kkt_reduction_ratio, problem
            ),
            lambda: (solver_state, last_restart_info),
        )

    def run_restart_scheme_feasibility_polishing(
        self,
        problem: QuadraticProgrammingProblem,
        current_solver_state: PdhgSolverState,
        restart_solver_state: PdhgSolverState,
        last_restart_info: RestartInfo,
    ):
        """
        Check restart criteria based on current and average KKT residuals.

        Parameters
        ----------
        problem : QuadraticProgrammingProblem
            The quadratic programming problem instance.
        solver_state : PdhgSolverState
            The current solver state.
        last_restart_info : RestartInfo
            Information from the last restart.

        Returns
        -------
        tuple
            The new solver state, and the new last restart info.
        """
        do_restart, kkt_reduction_ratio = cond(
            restart_solver_state.solutions_count == 0,
            lambda: (False, last_restart_info.reduction_ratio_last_trial),
            lambda: restart_criteria_met_fixed_point(
                self._restart_params, restart_solver_state, last_restart_info
            ),
        )
        return cond(
            do_restart,
            lambda: self.perform_restart(
                restart_solver_state, last_restart_info, kkt_reduction_ratio, problem
            ),
            lambda: (current_solver_state, last_restart_info),
        )

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
            The original problem and scaled problem data.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        tuple
            The updated solver state, the updated last restart info, whether to terminate, the scaled problem, and the cached quadratic programming information.
        """
        # Skip termination check for initial iterations
        restarted_solver_state, new_last_restart_info = self.run_restart_scheme(
            scaled_problem.scaled_qp, solver_state, last_restart_info
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
        # Check for termination
        should_terminate, termination_status, convergence_information = (
            check_termination_criteria(
                scaled_problem,
                solver_state,
                self._termination_criteria,
                qp_cache,
                solver_state.numerical_error,
                1.0,
                self.optimality_norm,
                False,
            )
        )

        restarted_solver_state, new_last_restart_info = self.run_restart_scheme(
            scaled_problem.scaled_qp, solver_state, last_restart_info
        )

        new_solver_state = self.take_multiple_steps(
            restarted_solver_state, scaled_problem.scaled_qp
        )
        new_solver_state.termination_status = termination_status
        return (
            new_solver_state,
            new_last_restart_info,
            should_terminate,
            scaled_problem,
            qp_cache,
            convergence_information,
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
            scaled_problem, solver_state, self._initial_primal_weight, average=False
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
        return new_solver_state.current_primal_solution, should_terminate

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
            self.run_restart_scheme_feasibility_polishing(
                primal_feasibility_problem.scaled_qp,
                primal_polishing_solver_state,
                zeroed_dual_solver_state,
                last_restart_info,
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
            average=False,
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
                scaled_problem, solver_state, self._initial_primal_weight, average=False
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
        return new_solver_state.current_dual_solution, should_terminate

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
            self.run_restart_scheme_feasibility_polishing(
                dual_feasibility_problem.scaled_qp,
                dual_polishing_solver_state,
                zeroed_primal_solver_state,
                last_restart_info,
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
            average=False,
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

        Returns
        -------
        SaddlePointOutput
            The solution to the optimization problem.
        """
        setup_logger(self.verbose, self.debug)
        # validate(original_problem)
        # config_check(params)
        self.check_config(original_problem.is_lp)
        qp_cache = cached_quadratic_program_info(original_problem, self.optimality_norm)

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
            scaled_problem, initial_primal_solution, initial_dual_solution
        )

        # Iteration loop
        display_iteration_stats_heading()

        iteration_start_time = timeit.default_timer()
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
                solver_state.current_primal_solution,
                solver_state.current_primal_product,
                solver_state.current_dual_solution,
                solver_state.current_dual_product,
                solver_state.current_primal_obj_product,
            ) = cond(
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
                    solver_state.current_primal_solution,
                    solver_state.current_primal_product,
                    solver_state.current_dual_solution,
                    solver_state.current_dual_product,
                    solver_state.current_primal_obj_product,
                ),
            )
            ci = compute_convergence_information(
                scaled_problem.original_qp,
                qp_cache,
                solver_state.current_primal_solution
                / scaled_problem.variable_rescaling,
                solver_state.current_dual_solution
                / scaled_problem.constraint_rescaling,
                self.abs_rel / self.rel_eps,
                solver_state.current_primal_product
                * scaled_problem.constraint_rescaling,
                solver_state.current_dual_product * scaled_problem.variable_rescaling,
                solver_state.current_primal_obj_product
                * scaled_problem.variable_rescaling,
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
            solver_state.current_primal_solution,
            solver_state.current_dual_solution,
            solver_state.num_iterations,
            solver_state.termination_status,
            timing,
            ci,
        )
        return unscaled_saddle_point_output(
            scaled_problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_solution,
            solver_state.termination_status,
            solver_state.num_iterations - 1,
            ci,
            timing,
        )
