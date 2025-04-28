from typing import Union

import jax
import jax.numpy as jnp

from mpax.iteration_stats_utils import evaluate_unscaled_iteration_stats
from mpax.utils import (
    CachedQuadraticProgramInfo,
    ConvergenceInformation,
    QuadraticProgrammingProblem,
    TerminationCriteria,
    TerminationStatus,
    PdhgSolverState,
    ScaledQpProblem,
)


def validate_termination_criteria(criteria: TerminationCriteria) -> None:
    """
    Validates the termination criteria to ensure all parameters are within acceptable ranges.

    Parameters
    ----------
    criteria : TerminationCriteria
        The criteria to validate.

    Raises
    ------
    ValueError
        If any of the criteria parameters are not valid.
    """
    if criteria.eps_primal_infeasible < 0:
        raise ValueError("eps_primal_infeasible must be nonnegative.")
    if criteria.eps_dual_infeasible < 0:
        raise ValueError("eps_dual_infeasible must be nonnegative.")
    # if criteria.time_sec_limit <= 0:
    #     raise ValueError("time_sec_limit must be positive.")
    if criteria.iteration_limit <= 0:
        raise ValueError("iteration_limit must be positive.")


def cached_quadratic_program_info(
    qp: QuadraticProgrammingProblem, norm_ord: float
) -> CachedQuadraticProgramInfo:
    """
    Computes information about the quadratic program used in termination criteria.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem instance.

    Returns
    -------
    CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    """
    return CachedQuadraticProgramInfo(
        jnp.linalg.norm(qp.objective_vector, ord=norm_ord),
        jnp.linalg.norm(qp.right_hand_side, ord=norm_ord),
    )


def optimality_criteria_met(
    rel_tol: float, convergence_information: ConvergenceInformation
) -> bool:
    """
    Checks if the algorithm should terminate declaring the optimal solution is found.

    Parameters
    ----------
    rel_tol : float
        Relative tolerance.
    convergence_information : ConvergenceInformation
        Convergence information of the current iteration.

    Returns
    -------
    bool
        True if optimality criteria are met, False otherwise.
    """
    ci = convergence_information
    gap = ci.relative_optimality_gap
    dual_err = ci.relative_dual_residual_norm
    primal_err = ci.relative_primal_residual_norm
    return (dual_err < rel_tol) & (primal_err < rel_tol) & (gap < rel_tol)


def primal_infeasibility_criteria_met(
    eps_primal_infeasible: float, infeasibility_information
) -> bool:
    """
    Checks if the algorithm should terminate declaring the primal is infeasible.

    Parameters
    ----------
    eps_primal_infeasible : float
        The tolerance for primal infeasibility.
    infeasibility_information : InfeasibilityInformation
        Information regarding infeasibility.

    Returns
    -------
    bool
        True if primal infeasibility criteria are met, False otherwise.
    """
    ii = infeasibility_information
    return jax.lax.cond(
        ii.dual_ray_objective <= 0.0,
        lambda _: False,
        lambda _: ii.max_dual_ray_infeasibility / ii.dual_ray_objective
        <= eps_primal_infeasible,
        operand=None,
    )


def dual_infeasibility_criteria_met(
    eps_dual_infeasible: float, infeasibility_information
) -> bool:
    """
    Checks if the algorithm should terminate declaring the dual is infeasible.

    Parameters
    ----------
    eps_dual_infeasible : float
        The tolerance for dual infeasibility.
    infeasibility_information : InfeasibilityInformation
        Information regarding infeasibility.

    Returns
    -------
    bool
        True if dual infeasibility criteria are met, False otherwise.
    """
    ii = infeasibility_information
    return jax.lax.cond(
        ii.primal_ray_linear_objective >= 0.0,
        lambda _: False,
        lambda _: ii.max_primal_ray_infeasibility / (-ii.primal_ray_linear_objective)
        <= eps_dual_infeasible,
        operand=None,
    )


def check_termination_criteria(
    scaled_problem,
    solver_state,
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    numerical_error: bool,
    elapsed_time: float,
    norm_ord: float,
    average: bool = True,
    infeasibility_detection: bool = True,
) -> Union[str, bool]:
    """
    Checks if the given iteration_stats satisfy the termination criteria.

    Parameters
    ----------
    criteria : TerminationCriteria
        Termination criteria to check against.
    qp_cache : CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    iteration_stats : IterationStats
        Statistics of the current iteration.

    Returns
    -------
    Union[str, bool]
        Termination reason if criteria are met, False otherwise.
    """
    eps_ratio = criteria.eps_abs / criteria.eps_rel
    current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        solver_state,
        elapsed_time,
        eps_ratio,
        norm_ord,
        average,
        infeasibility_detection,
    )
    should_terminate = False
    termination_status = TerminationStatus.UNSPECIFIED
    should_terminate, termination_status = jax.lax.cond(
        optimality_criteria_met(
            criteria.eps_rel, current_iteration_stats.convergence_information
        ),
        lambda: (True, TerminationStatus.OPTIMAL),
        lambda: (should_terminate, termination_status),
    )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False)
        & primal_infeasibility_criteria_met(
            criteria.eps_primal_infeasible,
            current_iteration_stats.infeasibility_information,
        ),
        lambda: (True, TerminationStatus.PRIMAL_INFEASIBLE),
        lambda: (should_terminate, termination_status),
    )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False)
        & dual_infeasibility_criteria_met(
            criteria.eps_dual_infeasible,
            current_iteration_stats.infeasibility_information,
        ),
        lambda: (True, TerminationStatus.DUAL_INFEASIBLE),
        lambda: (should_terminate, termination_status),
    )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False)
        & (current_iteration_stats.iteration_number >= criteria.iteration_limit),
        lambda: (True, TerminationStatus.ITERATION_LIMIT),
        lambda: (should_terminate, termination_status),
    )

    # should_terminate, termination_status = jax.lax.cond(
    #     (should_terminate == False) & (elapsed_time >= criteria.time_sec_limit),
    #     lambda _: (True, TerminationStatus.TIME_LIMIT),
    #     lambda _: (should_terminate, termination_status),
    #     operand=None,
    # )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False) & numerical_error,
        lambda _: (True, TerminationStatus.NUMERICAL_ERROR),
        lambda _: (should_terminate, termination_status),
        operand=None,
    )
    # should_terminate, termination_status = jax.lax.cond(
    #     (should_terminate == True)
    #     & ((termination_status == TerminationStatus.PRIMAL_INFEASIBLE) | (termination_status == TerminationStatus.DUAL_INFEASIBLE)),
    #     lambda _: (False, TerminationStatus.UNSPECIFIED),
    #     lambda _: (should_terminate, termination_status),
    #     operand=None,
    # )
    return (
        should_terminate,
        termination_status,
        current_iteration_stats.convergence_information,
    )


def check_primal_feasibility(
    scaled_problem: ScaledQpProblem,
    solver_state: PdhgSolverState,
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    elapsed_time: float,
    norm_ord: float,
    average: bool = True,
) -> bool:
    """
    Checks if the given iteration_stats satisfy the termination criteria.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        The scaled quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    criteria : TerminationCriteria
        Termination criteria to check against.
    qp_cache : CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    elapsed_time : float
        Elapsed time since the start of the algorithm.
    norm_ord : float
        Order of the norm.
    average : bool, optional
        Whether is raPDHG, by default True.

    Returns
    -------
    bool
        True if primal feasibility criteria are met, False otherwise.
    """
    eps_ratio = criteria.eps_abs / criteria.eps_rel
    current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        solver_state,
        elapsed_time,
        eps_ratio,
        norm_ord,
        average,
        infeasibility_detection=False,
    )
    ci = current_iteration_stats.convergence_information
    return ci.relative_primal_residual_norm < criteria.eps_rel


def check_dual_feasibility(
    scaled_problem: ScaledQpProblem,
    solver_state: PdhgSolverState,
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    elapsed_time: float,
    norm_ord: float,
    average: bool = True,
) -> Union[str, bool]:
    """
    Checks if the given iteration_stats satisfy the termination criteria.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        The scaled quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    criteria : TerminationCriteria
        Termination criteria to check against.
    qp_cache : CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    elapsed_time : float
        Elapsed time since the start of the algorithm.
    norm_ord : float
        Order of the norm.
    average : bool, optional
        Whether is raPDHG, by default True.

    Returns
    -------
    bool
        True if dual feasibility criteria are met, False otherwise.
    """
    eps_ratio = criteria.eps_abs / criteria.eps_rel
    current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        solver_state,
        elapsed_time,
        eps_ratio,
        norm_ord,
        average,
        infeasibility_detection=False,
    )
    ci = current_iteration_stats.convergence_information
    return ci.relative_dual_residual_norm < criteria.eps_rel
