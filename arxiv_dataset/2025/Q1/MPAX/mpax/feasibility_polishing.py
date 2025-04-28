from jax.lax import cond
import jax.numpy as jnp
from mpax.utils import ScaledQpProblem, RestartInfo
from mpax.restart import select_initial_primal_weight


def init_primal_feasibility_polishing(
    scaled_problem, solver_state, initial_primal_weight, average=True
):
    zero_objective_matrix = jnp.zeros_like(scaled_problem.original_qp.objective_matrix)
    zero_objective_vector = jnp.zeros_like(scaled_problem.original_qp.objective_vector)
    primal_feasibility_problem = ScaledQpProblem(
        original_qp=scaled_problem.original_qp.replace(
            objective_matrix=zero_objective_matrix,
            objective_vector=zero_objective_vector,
            objective_constant=0,
        ),
        scaled_qp=scaled_problem.scaled_qp.replace(
            objective_matrix=zero_objective_matrix,
            objective_vector=zero_objective_vector,
            objective_constant=0,
        ),
        constraint_rescaling=scaled_problem.constraint_rescaling,
        variable_rescaling=scaled_problem.variable_rescaling,
    )
    zero_dual_solution = jnp.zeros_like(solver_state.current_dual_solution)
    zero_dual_product = jnp.zeros_like(solver_state.current_dual_product)
    primal_solution = cond(
        average == True,
        lambda: solver_state.avg_primal_solution,
        lambda: solver_state.current_primal_solution,
    )
    primal_product = cond(
        average == True,
        lambda: solver_state.avg_primal_product,
        lambda: solver_state.current_primal_product,
    )
    primal_feasibility_solver_state = solver_state.replace(
        current_primal_solution=primal_solution,
        current_primal_product=primal_product,
        current_dual_solution=zero_dual_solution,
        current_dual_product=zero_dual_product,
        primal_weight=initial_primal_weight,
        num_iterations=0,
        num_steps_tried=0,
        weights_sum=0.0,
        solutions_count=0,
        avg_dual_solution=zero_dual_solution,
        avg_dual_product=zero_dual_product,
        initial_primal_solution=solver_state.current_primal_solution,
        initial_primal_product=solver_state.current_primal_product,
        initial_dual_solution=zero_dual_solution,
        initial_dual_product=zero_dual_product,
        delta_dual=zero_dual_solution,
    )
    last_restart_info = RestartInfo(
        primal_solution=primal_solution,
        dual_solution=jnp.zeros_like(solver_state.current_dual_solution),
        primal_diff=jnp.zeros_like(solver_state.current_primal_solution),
        dual_diff=jnp.zeros_like(solver_state.current_dual_solution),
        primal_diff_product=jnp.zeros_like(solver_state.current_primal_product),
        primal_product=primal_product,
        dual_product=jnp.zeros_like(solver_state.current_dual_product),
    )
    return (
        primal_feasibility_problem,
        primal_feasibility_solver_state,
        last_restart_info,
    )


def set_dual_solution_to_zero(solver_state):
    zero_dual_solution = jnp.zeros_like(solver_state.current_dual_solution)
    zero_dual_product = jnp.zeros_like(solver_state.current_dual_product)
    zeroed_dual_solver_state = solver_state.replace(
        current_dual_solution=zero_dual_solution,
        current_dual_product=zero_dual_product,
        avg_dual_solution=zero_dual_solution,
        avg_dual_product=zero_dual_product,
        initial_dual_solution=zero_dual_solution,
        initial_dual_product=zero_dual_product,
    )
    return zeroed_dual_solver_state


def init_dual_feasibility_polishing(
    scaled_problem, solver_state, initial_primal_weight, average=True
):
    dual_feasibility_problem = ScaledQpProblem(
        original_qp=scaled_problem.original_qp.replace(
            variable_lower_bound=jnp.where(
                scaled_problem.original_qp.isfinite_variable_lower_bound,
                0.0,
                scaled_problem.original_qp.variable_lower_bound,
            ),
            variable_upper_bound=jnp.where(
                scaled_problem.original_qp.isfinite_variable_upper_bound,
                0.0,
                scaled_problem.original_qp.variable_upper_bound,
            ),
            right_hand_side=jnp.zeros_like(scaled_problem.original_qp.right_hand_side),
        ),
        scaled_qp=scaled_problem.scaled_qp.replace(
            variable_lower_bound=jnp.where(
                scaled_problem.scaled_qp.isfinite_variable_lower_bound,
                0.0,
                scaled_problem.scaled_qp.variable_lower_bound,
            ),
            variable_upper_bound=jnp.where(
                scaled_problem.scaled_qp.isfinite_variable_upper_bound,
                0.0,
                scaled_problem.scaled_qp.variable_upper_bound,
            ),
            right_hand_side=jnp.zeros_like(scaled_problem.scaled_qp.right_hand_side),
        ),
        constraint_rescaling=scaled_problem.constraint_rescaling,
        variable_rescaling=scaled_problem.variable_rescaling,
    )
    # Check for termination
    zero_primal_solution = jnp.zeros_like(solver_state.current_primal_solution)
    zero_primal_product = jnp.zeros_like(solver_state.current_primal_product)
    dual_solution = cond(
        average == True,
        lambda: solver_state.avg_dual_solution,
        lambda: solver_state.current_dual_solution,
    )
    dual_product = cond(
        average == True,
        lambda: solver_state.avg_dual_product,
        lambda: solver_state.current_dual_product,
    )
    dual_feasibility_solver_state = solver_state.replace(
        current_primal_solution=zero_primal_solution,
        current_primal_product=zero_primal_product,
        current_dual_solution=dual_solution,
        current_dual_product=dual_product,
        primal_weight=initial_primal_weight,
        num_iterations=0,
        num_steps_tried=0,
        weights_sum=0.0,
        solutions_count=0,
        avg_primal_solution=zero_primal_solution,
        avg_primal_product=zero_primal_product,
        initial_primal_solution=zero_primal_solution,
        initial_primal_product=zero_primal_product,
        initial_dual_solution=solver_state.current_dual_solution,
        initial_dual_product=solver_state.current_dual_product,
        delta_primal=zero_primal_solution,
        delta_primal_product=zero_primal_product,
    )
    last_restart_info = RestartInfo(
        primal_solution=jnp.zeros_like(solver_state.avg_primal_solution),
        dual_solution=dual_solution,
        primal_diff=jnp.zeros_like(solver_state.current_primal_solution),
        dual_diff=jnp.zeros_like(solver_state.current_dual_solution),
        primal_diff_product=jnp.zeros_like(solver_state.current_primal_product),
        primal_product=jnp.zeros_like(solver_state.current_primal_product),
        dual_product=dual_product,
    )

    return dual_feasibility_problem, dual_feasibility_solver_state, last_restart_info


def set_primal_solution_to_zero(solver_state):
    zero_primal_solution = jnp.zeros_like(solver_state.current_primal_solution)
    zero_primal_product = jnp.zeros_like(solver_state.current_primal_product)
    zeroed_primal_solver_state = solver_state.replace(
        current_primal_solution=zero_primal_solution,
        current_primal_product=zero_primal_product,
        avg_primal_solution=zero_primal_solution,
        avg_primal_product=zero_primal_product,
        initial_primal_solution=zero_primal_solution,
        initial_primal_product=zero_primal_product,
    )
    return zeroed_primal_solver_state
