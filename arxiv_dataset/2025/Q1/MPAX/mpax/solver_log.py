import logging
from typing import Union

from jax import debug, lax
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, BCSR

from mpax.utils import QuadraticProgrammingProblem, TerminationStatus, TimingData

logger = logging.getLogger(__name__)


def setup_logger(verbose: bool = False, debug: bool = False):
    """Configures logging for the entire application."""
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()  # Clear any existing handlers

    # Set logging level based on verbose parameter
    if debug:
        logging_level = logging.DEBUG
        logging.getLogger("jax").setLevel(logging.INFO)
        logging.getLogger("jaxlib").setLevel(logging.INFO)
        logging.getLogger("xla").setLevel(logging.INFO)
    elif verbose:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING
    logger.setLevel(logging_level)

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    # Add the handler to the logger
    logger.addHandler(ch)


def jax_debug_log(
    fmt: str, *args, logger: logging.Logger = None, level: int = logging.DEBUG, **kwargs
):
    """Logs a message using a specified logger and logging level."""

    # Use the provided logger, or default to the root logger
    if logger is None:
        logger = logging.getLogger()

    # Wrap the logging call inside jax.debug.callback
    debug.callback(
        lambda *a, **k: logger.log(
            level, fmt.format(*a, **k)
        ),  # Log with specified level
        *args,
        ordered=False,
        **kwargs,
    )


def get_row_l2_norms(matrix: Union[BCOO, BCSR, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the L2 norms of the rows of a sparse BCOO matrix.

    Parameters
    ----------
    matrix : Union[BCOO, BCSR, jnp.ndarray]
        Sparse matrix in BCOO, BCSR, jnp.ndarray format.

    Returns
    -------
    jnp.ndarray
        The L2 norms of the rows of the matrix.
    """
    if isinstance(matrix, (BCOO, BCSR)):
        row_norms = jnp.zeros(matrix.shape[0])
        nzval = matrix.data
        if isinstance(matrix, BCSR):
            rowval = matrix.to_bcoo().indices[:, 0]
        elif isinstance(matrix, BCOO):
            rowval = matrix.indices[:, 0]
        # Accumulate the sum of squares for each row
        row_norms = jnp.sqrt(row_norms.at[rowval].add(nzval**2))
    elif isinstance(matrix, jnp.ndarray):
        row_norms = jnp.linalg.norm(matrix, ord=2, axis=1)

    # Return the square root to get the L2 norms
    return row_norms


def get_col_l2_norms(matrix: Union[BCOO, BCSR, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the L2 norms of the columns of a sparse BCOO matrix.

    Parameters
    ----------
    matrix : Union[BCOO, BCSR, jnp.ndarray]
        Sparse matrix in BCOO, BCSR, jnp.ndarray format.

    Returns
    -------
    jnp.ndarray
        The L2 norms of the columns of the matrix.
    """
    if isinstance(matrix, (BCOO, BCSR)):
        col_norms = jnp.zeros(matrix.shape[1])
        nzval = matrix.data
        if isinstance(matrix, BCSR):
            colval = matrix.to_bcoo().indices[:, 1]
        elif isinstance(matrix, BCOO):
            colval = matrix.indices[:, 1]

        # Accumulate the sum of squares for each column
        col_norms = jnp.sqrt(col_norms.at[colval].add(nzval**2))
    elif isinstance(matrix, jnp.ndarray):
        col_norms = jnp.linalg.norm(matrix, ord=2, axis=0)

    # Return the square root to get the L2 norms
    return col_norms


def get_row_l_inf_norms(matrix: Union[BCOO, BCSR, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the infinity norms (L-infinity) of the rows of a sparse BCOO matrix.

    Parameters
    ----------
    matrix : Union[BCOO, BCSR, jnp.ndarray]
        Sparse matrix in BCOO, BCSR, jnp.ndarray format.

    Returns
    -------
    jnp.ndarray
        The L-infinity norms of the rows of the matrix.
    """
    if isinstance(matrix, (BCOO, BCSR)):
        row_norms = jnp.zeros(matrix.shape[0])
        nzval = matrix.data
        if isinstance(matrix, BCSR):
            rowval = matrix.to_bcoo().indices[:, 0]
        elif isinstance(matrix, BCOO):
            rowval = matrix.indices[:, 0]
        # Accumulate the max absolute value for each row
        row_norms = row_norms.at[rowval].max(jnp.abs(nzval))
    elif isinstance(matrix, jnp.ndarray):
        row_norms = jnp.linalg.norm(matrix, ord=jnp.inf, axis=1)

    return row_norms


def get_col_l_inf_norms(matrix: Union[BCOO, BCSR, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the infinity norms (L-infinity) of the columns of a sparse BCOO matrix.

    Parameters
    ----------
    matrix : Union[BCOO, BCSR, jnp.ndarray]
        Sparse matrix in BCOO, BCSR, jnp.ndarray format.

    Returns
    -------
    jnp.ndarray
        The L-infinity norms of the columns of the matrix.
    """
    if isinstance(matrix, (BCOO, BCSR)):
        col_norms = jnp.zeros(matrix.shape[1])
        nzval = matrix.data
        if isinstance(matrix, BCSR):
            colval = matrix.to_bcoo().indices[:, 1]
        else:
            colval = matrix.indices[:, 1]
        # Accumulate the max absolute value for each column
        col_norms = col_norms.at[colval].max(jnp.abs(nzval))
    elif isinstance(matrix, jnp.ndarray):
        col_norms = jnp.linalg.norm(matrix, ord=jnp.inf, axis=0)

    return col_norms


def display_problem_details(qp: QuadraticProgrammingProblem) -> None:
    """
    Display the details of the quadratic programming problem, including the size of the problem,
    and norms of the constraint matrix and objective vectors.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem object containing the matrix and vector details.
    """
    if logging.root.level == logging.INFO:
        jax_debug_log(
            "There are {:d} variables, {:d} constraints (including {:d} equalities) and {:d} nonzero coefficients.",
            qp.constraint_matrix.shape[1],
            qp.constraint_matrix.shape[0],
            qp.num_equalities,
            len(qp.constraint_matrix.data),
            logger=logger,
            level=logging.INFO,
        )

        nz_constraints = qp.constraint_matrix.data
        jax_debug_log(
            "Absolute value of nonzero constraint matrix elements:\n"
            "  largest={:.6f}, smallest={:.6f}, avg={:.6f}",
            jnp.max(jnp.abs(nz_constraints)),
            jnp.min(jnp.abs(nz_constraints)),
            jnp.mean(jnp.abs(nz_constraints)),
            logger=logger,
            level=logging.INFO,
        )

        col_norms = get_col_l_inf_norms(qp.constraint_matrix)
        row_norms = get_row_l_inf_norms(qp.constraint_matrix)
        jax_debug_log(
            "Constraint matrix, infinity norm:\n"
            "  max_col={:.6f}, min_col={:.6f}, max_row={:.6f}, min_row={:.6f}",
            jnp.max(col_norms),
            jnp.min(col_norms),
            jnp.max(row_norms),
            jnp.min(row_norms),
            logger=logger,
            level=logging.INFO,
        )

        if len(qp.objective_matrix.data) > 0:
            nz_objectives = qp.objective_matrix.data
            jax_debug_log(
                "Absolute value of objective matrix elements:"
                "  largest={:.6f}, smallest={:.6f}, avg={:.6f}",
                jnp.max(jnp.abs(nz_objectives)),
                jnp.min(jnp.abs(nz_objectives)),
                jnp.mean(jnp.abs(nz_objectives)),
                logger=logger,
                level=logging.INFO,
            )

        jax_debug_log(
            "Absolute value of objective vector elements:\n"
            "  largest={:.6f}, smallest={:.6f}, avg={:.6f}",
            jnp.max(jnp.abs(qp.objective_vector)),
            jnp.min(jnp.abs(qp.objective_vector)),
            jnp.mean(jnp.abs(qp.objective_vector)),
            logger=logger,
            level=logging.INFO,
        )

        jax_debug_log(
            "Absolute value of rhs vector elements:\n"
            "  largest={:.6f}, smallest={:.6f}, avg={:.6f}",
            jnp.max(jnp.abs(qp.right_hand_side)),
            jnp.min(jnp.abs(qp.right_hand_side)),
            jnp.mean(jnp.abs(qp.right_hand_side)),
            logger=logger,
            level=logging.INFO,
        )

        bound_gaps = qp.variable_upper_bound - qp.variable_lower_bound
        finite_label = jnp.isfinite(bound_gaps)
        jax_debug_log(
            "Gap between upper and lower bounds:\n"
            "  # finite= {:d} of {:d}, largest= {:f}, smallest= {:f}, avg= {:f}",
            jnp.sum(finite_label),
            len(bound_gaps),
            jnp.max(bound_gaps, initial=jnp.nan, where=finite_label),
            jnp.min(bound_gaps, initial=jnp.nan, where=finite_label),
            jnp.mean(bound_gaps, where=finite_label),
            logger=logger,
            level=logging.INFO,
        )


def display_iteration_stats_heading():
    """
    Display the heading for the iteration statistics.
    """
    if logging.root.level == logging.INFO:
        jax_debug_log(
            "{:<7s} | {:<25s} | {:<25s} | {:<23s} |",
            "runtime",
            "residuals",
            "solution information",
            "relative residuals",
            logger=logger,
            level=logging.INFO,
        )
        jax_debug_log(
            "{:<7s} | {:<8s} {:<8s} {:<7s} | {:<8s} {:<8s} {:<7s} | {:<7s} {:<7s} {:<7s} |",
            "#iter",
            # "seconds",
            "pr_norm",
            "du_norm",
            "gap",
            "pr_obj",
            "pr_norm",
            "du_norm",
            "rel_pr",
            "rel_du",
            "rel_gap",
            logger=logger,
            level=logging.INFO,
        )
    elif logging.root.level == logging.DEBUG:
        jax_debug_log(
            "{:<7s} | {:<25s} | {:<25s} | {:<23s} | {:<16s} | {:<17s} |",
            "runtime",
            "residuals",
            "solution information",
            "relative residuals",
            "primal ray",
            "dual ray",
            logger=logger,
            level=logging.DEBUG,
        )
        jax_debug_log(
            "{:<7s} | {:<8s} {:<8s} {:<7s} | {:<8s} {:<8s} {:<7s} | {:<7s} {:<7s} {:<7s} | {:<8s} {:<7s} | {:<8s} {:<8s} |",
            "#iter",
            # "seconds",
            "pr_norm",
            "du_norm",
            "gap",
            "pr_obj",
            "pr_norm",
            "du_norm",
            "rel_pr",
            "rel_du",
            "rel_gap",
            "pr_norm",
            "linear",
            "du_norm",
            "dual_obj",
            logger=logger,
            level=logging.DEBUG,
        )


def display_iteration_stats(stats, solver_state):
    """
    Display the iteration statistics.

    Parameters
    ----------
    stats : IterationStats
        The iteration statistics to be displayed.
    solver_state : SolverState
        The current state of the solver.
    """
    conv_info = stats.convergence_information
    infeas_info = stats.infeasibility_information
    if logging.root.level == logging.DEBUG:
        jax_debug_log(
            "{:-7d} | {:.1e}  {:.1e}  {:.1e} | {:.1e}  {:.1e}  {:.1e} | {:.1e} {:.1e} {:.1e} | {:.1e}  {:.1e} | {:.1e}  {:.1e}  |",
            stats.iteration_number,
            # stats.cumulative_time_sec,
            conv_info.primal_residual_norm,
            conv_info.dual_residual_norm,
            conv_info.absolute_optimality_gap,
            conv_info.primal_objective,
            conv_info.primal_solution_norm,
            conv_info.dual_solution_norm,
            conv_info.relative_primal_residual_norm,
            conv_info.relative_dual_residual_norm,
            conv_info.relative_optimality_gap,
            infeas_info.max_primal_ray_infeasibility,
            infeas_info.primal_ray_linear_objective,
            infeas_info.max_dual_ray_infeasibility,
            infeas_info.dual_ray_objective,
            logger=logger,
            level=logging.DEBUG,
        )
        jax_debug_log(
            "Iteration {iteration}: norms=({norm_primal:.2e}, {norm_dual:.2e}), inv_step_size={inv_step_size:.2e}",
            iteration=solver_state.num_iterations,
            norm_primal=jnp.linalg.norm(solver_state.current_primal_solution),
            norm_dual=jnp.linalg.norm(solver_state.current_dual_solution),
            inv_step_size=1 / solver_state.step_size,
            logger=logger,
            level=logging.DEBUG,
        )
        jax_debug_log(
            "primal_weight={primal_weight:.4e}",
            primal_weight=solver_state.primal_weight,
            logger=logger,
            level=logging.DEBUG,
        )
    elif logging.root.level == logging.INFO:
        jax_debug_log(
            "{:-7d} | {:.1e}  {:.1e}  {:.1e} | {:.1e}  {:.1e}  {:.1e} | {:.1e} {:.1e} {:.1e} |",
            stats.iteration_number,
            # stats.cumulative_time_sec,
            conv_info.primal_residual_norm,
            conv_info.dual_residual_norm,
            conv_info.absolute_optimality_gap,
            conv_info.primal_objective,
            conv_info.primal_solution_norm,
            conv_info.dual_solution_norm,
            conv_info.relative_primal_residual_norm,
            conv_info.relative_dual_residual_norm,
            conv_info.relative_optimality_gap,
            logger=logger,
            level=logging.INFO,
        )


def pdhg_final_log(
    problem: QuadraticProgrammingProblem,
    avg_primal_solution: jnp.ndarray,
    avg_dual_solution: jnp.ndarray,
    iteration: int,
    termination_status: TerminationStatus,
    timing: TimingData,
    convergence_information,
):
    """
    Logs the final statistics and results of the PDHG algorithm.

    Parameters
    ----------
    avg_primal_solution : jax.numpy.ndarray
        The averaged primal solution.
    avg_dual_solution : jax.numpy.ndarray
        The averaged dual solution.
    iteration : int
        The current iteration count.
    termination_status : TerminationStatus
        The reason for termination.
    last_iteration_stats : IterationStats
        Statistics from the last iteration.
    timing : TimingData
        Timing information.
    """
    if logging.root.level <= logging.INFO:
        # Print primal norms
        jax_debug_log(
            "  primal norms: L1={:15.10g}, L2={:15.10g}, Linf={:15.10g}",
            jnp.linalg.norm(avg_primal_solution, 1),
            jnp.linalg.norm(avg_primal_solution),
            jnp.linalg.norm(avg_primal_solution, jnp.inf),
            logger=logger,
            level=logging.INFO,
        )
        # Print dual norms
        jax_debug_log(
            "  dual norms:   L1={:15.10g}, L2={:15.10g}, Linf={:15.10g}",
            jnp.linalg.norm(avg_dual_solution, 1),
            jnp.linalg.norm(avg_dual_solution),
            jnp.linalg.norm(avg_dual_solution, jnp.inf),
            logger=logger,
            level=logging.INFO,
        )
        jax_debug_log(
            "Terminated after {:d} iterations. Termination Status: {:d}",
            iteration,
            termination_status,
            logger=logger,
            level=logging.INFO,
        )
        jax_debug_log(
            " - Primal objective: {:.10e}\n"
            " - Dual objective: {:.10e}\n"
            " - Corrected dual objective: {:.10e}\n"
            " - Abs gap: {:.5e}\n"
            " - Rel gap: {:.5e}\n"
            " - Abs Primal residual: {:.5e}\n"
            " - Abs Dual residual: {:.5e}\n"
            " - Rel primal residual: {:.5e}\n"
            " - Rel dual residual: {:.5e}",
            convergence_information.primal_objective,
            convergence_information.dual_objective,
            convergence_information.corrected_dual_objective,
            convergence_information.absolute_optimality_gap,
            convergence_information.relative_optimality_gap,
            convergence_information.primal_residual_norm,
            convergence_information.dual_residual_norm,
            convergence_information.relative_primal_residual_norm,
            convergence_information.relative_dual_residual_norm,
            logger=logger,
            level=logging.INFO,
        )

        # Log timing information with formatted string
        formatted_timing = "\n".join(
            f" - {key}: {value:.4f} seconds" for key, value in timing.items()
        )
        jax_debug_log(
            "Timing Information:\n{}\n"
            "(Using 'jax.jit(optimize)' directly may lead to inaccurate timing due to JAX's tracing and compilation process.)",
            formatted_timing,
            logger=logger,
            level=logging.INFO,
        )

    if logging.root.level <= logging.DEBUG:
        row_norms = get_row_l2_norms(problem.constraint_matrix)
        constraint_hardness = row_norms * jnp.abs(avg_dual_solution)
        col_norms = get_col_l2_norms(problem.constraint_matrix)
        variable_hardness = col_norms * jnp.abs(avg_primal_solution)
        jax_debug_log(
            "Constraint hardness: median_hardness={:f}, mean_hardness={:f}, quantile_99={:f}, hardest={:f}",
            jnp.median(constraint_hardness),
            jnp.mean(constraint_hardness),
            jnp.quantile(constraint_hardness, 0.99),
            jnp.max(constraint_hardness),
            logger=logger,
            level=logging.DEBUG,
        )
        jax_debug_log(
            "Variable hardness: median_hardness={:f}, mean_hardness={:f}, quantile_99={:f}, hardest={:f}",
            jnp.median(variable_hardness),
            jnp.mean(variable_hardness),
            jnp.quantile(variable_hardness, 0.99),
            jnp.max(variable_hardness),
            logger=logger,
            level=logging.DEBUG,
        )
