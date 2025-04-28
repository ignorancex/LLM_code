import logging
from copy import deepcopy
from typing import List, Tuple, Union

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR, bcoo_concatenate

from mpax.solver_log import (
    display_problem_details,
    get_col_l_inf_norms,
    get_row_l_inf_norms,
    get_row_l2_norms,
    get_col_l2_norms,
)
from mpax.utils import PresolveInfo, QuadraticProgrammingProblem, ScaledQpProblem

logger = logging.getLogger(__name__)


def validate(p: QuadraticProgrammingProblem) -> bool:
    """
    Check that the QuadraticProgrammingProblem is valid.

    Parameters
    ----------
    p : QuadraticProgrammingProblem
        The quadratic programming problem to validate.

    Returns
    -------
    bool
        True if the problem is valid, otherwise raises an error.
    """
    error_found = False

    if len(p.variable_lower_bound) != len(p.variable_upper_bound):
        logger.error(
            "%d != %d", len(p.variable_lower_bound), len(p.variable_upper_bound)
        )
        error_found = True

    if len(p.variable_lower_bound) != len(p.objective_vector):
        logger.error("%d != %d", len(p.variable_lower_bound), len(p.objective_vector))
        error_found = True

    if p.constraint_matrix.shape[0] != len(p.right_hand_side):
        logger.error("%d != %d", p.constraint_matrix.shape[0], len(p.right_hand_side))
        error_found = True

    if p.constraint_matrix.shape[1] != len(p.objective_vector):
        logger.error("%d != %d", p.constraint_matrix.shape[1], len(p.objective_vector))
        error_found = True

    if p.objective_matrix.shape != (len(p.objective_vector), len(p.objective_vector)):
        logger.error(
            "%s is not square with length %d",
            p.objective_matrix.shape,
            len(p.objective_vector),
        )
        error_found = True

    if jnp.any(p.variable_lower_bound == jnp.inf):
        logger.error(
            "sum(p.variable_lower_bound == Inf) = %s",
            jnp.sum(jnp.isinf(p.variable_lower_bound)),
        )
        error_found = True

    if jnp.any(p.variable_upper_bound == -jnp.inf):
        logger.error(
            "sum(p.variable_upper_bound == -Inf) = %s",
            jnp.sum(jnp.isinf(p.variable_upper_bound)),
        )
        error_found = True

    if jnp.any(jnp.isnan(p.variable_lower_bound)) or jnp.any(
        jnp.isnan(p.variable_upper_bound)
    ):
        logger.error("NaN found in variable bounds of QuadraticProgrammingProblem.")
        error_found = True

    if jnp.any(jnp.isinf(p.right_hand_side)) or jnp.any(jnp.isnan(p.right_hand_side)):
        logger.error(
            "NaN or Inf found in right hand side of QuadraticProgrammingProblem."
        )
        error_found = True

    if jnp.any(jnp.isinf(p.objective_vector)) or jnp.any(jnp.isnan(p.objective_vector)):
        logger.error(
            "NaN or Inf found in objective vector of QuadraticProgrammingProblem."
        )
        error_found = True

    if jnp.any(jnp.isinf(p.constraint_matrix.data)) or jnp.any(
        jnp.isnan(p.constraint_matrix.data)
    ):
        logger.error(
            "NaN or Inf found in constraint matrix of QuadraticProgrammingProblem."
        )
        error_found = True

    if jnp.any(jnp.isinf(p.objective_matrix.data)) or jnp.any(
        jnp.isnan(p.objective_matrix.data)
    ):
        logger.error(
            "NaN or Inf found in objective matrix of QuadraticProgrammingProblem."
        )
        error_found = True

    if error_found:
        raise ValueError(
            "Error found when validating QuadraticProgrammingProblem. See log statements for details."
        )

    return True


def remove_empty_rows(problem: QuadraticProgrammingProblem) -> List[int]:
    """
    Removes the empty rows of a quadratic programming problem.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified to store the transformed problem.

    Returns
    -------
    List[int]
        List of indices of the removed empty columns.
    """
    num_rows = problem.constraint_matrix.shape[0]

    if isinstance(problem.constraint_matrix, BCOO):
        seen_row = jnp.zeros(num_rows, dtype=bool)
        row_indices, _ = problem.constraint_matrix.indices.T

        for row in jnp.unique(row_indices):
            seen_row = seen_row.at[row].set(True)
    elif isinstance(problem.constraint_matrix, jnp.ndarray):
        seen_row = ~jnp.all(problem.constraint_matrix == 0, axis=1)
    else:
        raise TypeError("constraint_matrix must be either BCOO or jnp.ndarray.")

    empty_rows = jnp.where(~seen_row)[0]

    for row in empty_rows:
        if row > problem.num_equalities and problem.right_hand_side[row] > 0.0:
            raise ValueError("The problem is infeasible.")
        elif row <= problem.num_equalities and problem.right_hand_side[row] != 0.0:
            raise ValueError("The problem is infeasible.")

    if len(empty_rows) > 0:
        # Filter to keep only non-empty rows
        if isinstance(problem.constraint_matrix, BCOO):
            mask = jnp.isin(row_indices, jnp.where(seen_row)[0])
            new_data = problem.constraint_matrix.data[mask]
            new_indices = problem.constraint_matrix.indices[mask]

            # Create the filtered BCOO matrix
            new_coo_matrix = BCOO(
                (new_data, new_indices),
                shape=(jnp.sum(seen_row), problem.constraint_matrix.shape[1]),
            )

            # Assign back to the problem, using BCOO directly
            problem.constraint_matrix = new_coo_matrix
        elif isinstance(problem.constraint_matrix, jnp.ndarray):
            problem.constraint_matrix = problem.constraint_matrix[seen_row, :]
        problem.right_hand_side = problem.right_hand_side[seen_row]
        num_empty_equalities = jnp.sum(empty_rows <= problem.num_equalities)
        problem.num_equalities -= num_empty_equalities

    return empty_rows.tolist()


def remove_empty_columns(problem: QuadraticProgrammingProblem) -> List[int]:
    """
    Removes the empty columns of a quadratic programming problem.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified to store the transformed problem.

    Returns
    -------
    List[int]
        List of indices of the removed empty columns.
    """
    # Convert to BCOO format for easier manipulation
    num_columns = problem.constraint_matrix.shape[1]

    # Determine empty columns by checking if each column index appears
    if isinstance(problem.constraint_matrix, BCOO):
        col_indices = problem.constraint_matrix.indices[:, 1]  # Extract column indices
        is_empty_column = jnp.ones(num_columns, dtype=bool).at[col_indices].set(False)
    elif isinstance(problem.constraint_matrix, jnp.ndarray):
        is_empty_column = jnp.all(problem.constraint_matrix == 0, axis=0)
    else:
        raise TypeError("constraint_matrix must be either BCOO or jnp.ndarray.")

    empty_columns = jnp.where(is_empty_column)[0]
    # Filter non-empty columns and update problem matrices and vectors
    non_empty_mask = ~is_empty_column
    non_empty_columns = jnp.where(non_empty_mask)[0]

    # Update objective constant based on empty columns
    for col in empty_columns:
        objective_coef = problem.objective_vector[col]
        problem.objective_constant += (
            problem.variable_lower_bound[col] * objective_coef
            if objective_coef >= 0
            else problem.variable_upper_bound[col] * objective_coef
        )

    if isinstance(problem.constraint_matrix, BCOO):
        # Create a new BCOO matrix with non-empty columns only
        col_mapping = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(non_empty_columns.tolist())
        }
        mask = jnp.isin(col_indices, non_empty_columns)
        filtered_data = problem.constraint_matrix.data[mask]
        filtered_indices = jnp.array(
            [
                [row, col_mapping[int(col)]]
                for row, col in problem.constraint_matrix.indices[mask]
            ]
        )
        new_constraint_matrix = BCOO(
            (filtered_data, filtered_indices),
            shape=(problem.constraint_matrix.shape[0], len(non_empty_columns)),
        )
    else:
        new_constraint_matrix = problem.constraint_matrix[:, non_empty_mask]

    # Update problem attributes
    problem.constraint_matrix = new_constraint_matrix
    problem.objective_vector = problem.objective_vector[non_empty_mask]
    problem.variable_lower_bound = problem.variable_lower_bound[non_empty_mask]
    problem.variable_upper_bound = problem.variable_upper_bound[non_empty_mask]

    # Update objective_matrix (symmetric matrix)
    mask_rows = jnp.isin(problem.objective_matrix.indices[:, 0], non_empty_columns)
    mask_cols = jnp.isin(problem.objective_matrix.indices[:, 1], non_empty_columns)
    mask = mask_rows & mask_cols
    new_objective_data = problem.objective_matrix.data[mask]
    new_objective_indices = jnp.array(
        [
            [col_mapping[int(row)], col_mapping[int(col)]]
            for row, col in problem.objective_matrix.indices[mask]
        ]
    )
    problem.objective_matrix = BCOO(
        (new_objective_data, new_objective_indices),
        shape=(len(non_empty_columns), len(non_empty_columns)),
    )

    return empty_columns.tolist()


def transform_bounds_into_linear_constraints(qp: QuadraticProgrammingProblem) -> None:
    """
    Modifies the problem by transforming any finite variable bounds into linear constraints.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.
    """
    # Identify finite lower and upper bounds
    finite_lower_bound_indices = jnp.where(jnp.isfinite(qp.variable_lower_bound))[0]
    finite_upper_bound_indices = jnp.where(jnp.isfinite(qp.variable_upper_bound))[0]

    # Prepare the indices and non-zero values for the sparse identity block
    row_indices = jnp.arange(
        len(finite_lower_bound_indices) + len(finite_upper_bound_indices)
    )
    column_indices = jnp.concatenate(
        [finite_lower_bound_indices, finite_upper_bound_indices]
    )
    nonzeros = jnp.concatenate(
        [
            jnp.ones(len(finite_lower_bound_indices)),
            -jnp.ones(len(finite_upper_bound_indices)),
        ]
    )

    # Create a BCOO sparse identity block
    identity_block = BCOO(
        (nonzeros, jnp.vstack([row_indices, column_indices]).T),
        shape=(len(row_indices), len(qp.variable_lower_bound)),
    )

    # Update the constraint matrix with the new linear constraints
    qp.constraint_matrix = bcoo_concatenate(
        [qp.constraint_matrix, identity_block], dimension=0
    )

    # Update the right-hand side vector
    qp.right_hand_side = jnp.concatenate(
        [
            qp.right_hand_side,
            qp.variable_lower_bound[finite_lower_bound_indices],
            -qp.variable_upper_bound[finite_upper_bound_indices],
        ]
    )

    # Update variable bounds to be infinite
    qp.variable_lower_bound = jnp.full_like(qp.variable_lower_bound, -jnp.inf)
    qp.variable_upper_bound = jnp.full_like(qp.variable_upper_bound, jnp.inf)


def recover_original_solution(
    solution: jnp.ndarray, empty_indices: List[int], original_size: int
) -> jnp.ndarray:
    """
    Given a solution to the preprocessed problem this function recovers a solution to the original problem.

    Parameters
    ----------
    solution : jnp.ndarray
        The solution after preprocessing.
    empty_indices : List[int]
        Indices corresponding to portions of the solution that were eliminated in preprocessing.
    original_size : int
        Size of the solution vector before preprocessing.

    Returns
    -------
    jnp.ndarray
        The recovered solution to the original problem.
    """
    # Convert empty_indices to a JAX array for indexing
    empty_indices = jnp.array(empty_indices, dtype=jnp.int32)

    # Create a boolean mask indicating non-empty indices
    mask = jnp.ones(original_size, dtype=bool).at[empty_indices].set(False)

    # Initialize the original solution with zeros
    original_solution = jnp.zeros(original_size, dtype=solution.dtype)

    # Use scatter to place the solution values at the correct non-empty indices
    original_solution = original_solution.at[mask].set(solution)

    return original_solution


def undo_presolve(
    presolve_info: PresolveInfo,
    primal_solution: jnp.ndarray,
    dual_solution: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Recover the original problem's solution from the preprocessed problem's solution.

    Parameters
    ----------
    presolve_info : PresolveInfo
        Information allowing the presolve to be undone.
    primal_solution : jnp.ndarray
        The primal solution after preprocessing.
    dual_solution : jnp.ndarray
        The dual solution after preprocessing.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        The recovered primal and dual solutions to the original problem.
    """
    # Recover and clip the primal solution in one step
    recovered_primal_solution = recover_original_solution(
        primal_solution, presolve_info.empty_columns, presolve_info.original_primal_size
    )
    clipped_primal_solution = jnp.clip(
        recovered_primal_solution,
        presolve_info.variable_lower_bound,
        presolve_info.variable_upper_bound,
    )

    # Recover the dual solution
    recovered_dual_solution = recover_original_solution(
        dual_solution, presolve_info.empty_rows, presolve_info.original_dual_size
    )

    return clipped_primal_solution, recovered_dual_solution


def scale_problem(
    problem: QuadraticProgrammingProblem,
    constraint_rescaling: jnp.ndarray,
    variable_rescaling: jnp.ndarray,
) -> None:
    """
    Rescales `problem` in place.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.
    constraint_rescaling : jnp.ndarray
        The rescaling factors for the constraints.
    variable_rescaling : jnp.ndarray
        The rescaling factors for the variables.
    """
    # Scale the objective vector
    problem.objective_vector /= variable_rescaling

    # Scale the objective matrix using BCSR format directly
    if isinstance(problem.objective_matrix, jnp.ndarray):
        # Scale the matrix along the rows
        # variable_rescaling[:, None] reshapes variable_rescaling from (n,) to (n, 1),
        # enabling broadcasting along the rows. Each element in row i is divided by variable_rescaling[i].
        problem.objective_matrix = (
            problem.objective_matrix / variable_rescaling[:, None]
        )
        # Scale the matrix along the columns
        # variable_rescaling (with shape (n,)) is broadcasted along the columns,
        # so each element in column j is divided by variable_rescaling[j].
        problem.objective_matrix = problem.objective_matrix / variable_rescaling
    elif isinstance(problem.objective_matrix, BCOO):
        scaled_data = (
            problem.objective_matrix.data
            * (1.0 / variable_rescaling)[problem.objective_matrix.indices[:, 0]]
            * (1.0 / variable_rescaling)[problem.objective_matrix.indices[:, 1]]
        )
        problem.objective_matrix.data = scaled_data

    # Scale variable bounds
    problem.variable_upper_bound *= variable_rescaling
    problem.variable_lower_bound *= variable_rescaling

    # Scale the right-hand side vector
    problem.right_hand_side /= constraint_rescaling

    # Scale the constraint matrix
    if isinstance(problem.constraint_matrix, jnp.ndarray):
        problem.constraint_matrix = (
            problem.constraint_matrix / constraint_rescaling[:, None]
        )
        problem.constraint_matrix = problem.constraint_matrix / variable_rescaling
    elif isinstance(problem.constraint_matrix, BCOO):
        scaled_data = (
            problem.constraint_matrix.data
            * (1.0 / constraint_rescaling)[problem.constraint_matrix.indices[:, 0]]
            * (1.0 / variable_rescaling)[problem.constraint_matrix.indices[:, 1]]
        )
        problem.constraint_matrix.data = scaled_data
    problem.constraint_matrix_t = problem.constraint_matrix.T


def unscale_problem(
    problem: QuadraticProgrammingProblem,
    constraint_rescaling: jnp.ndarray,
    variable_rescaling: jnp.ndarray,
) -> None:
    """
    Recovers the original problem from the scaled problem.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.
    constraint_rescaling : jnp.ndarray
        The rescaling factors for the constraints.
    variable_rescaling : jnp.ndarray
        The rescaling factors for the variables.
    """
    scale_problem(problem, 1.0 / constraint_rescaling, 1.0 / variable_rescaling)


def l2_norm_rescaling(
    problem: QuadraticProgrammingProblem,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rescales a quadratic programming problem by dividing each row and column of the constraint matrix by the sqrt of its respective L2 norm.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple of vectors containing the row and column rescaling factors.
    """
    # Calculate L2 norms of rows and columns
    norm_of_rows = get_row_l2_norms(problem.constraint_matrix)
    norm_of_columns = jnp.sqrt(
        jnp.square(get_col_l2_norms(problem.constraint_matrix))
        + jnp.square(get_col_l2_norms(problem.objective_matrix))
    )
    # Avoid division by zero by setting norms to 1 where they are 0
    norm_of_rows = jnp.where(norm_of_rows == 0, 1.0, norm_of_rows)
    norm_of_columns = jnp.where(norm_of_columns == 0, 1.0, norm_of_columns)

    # Compute the rescaling factors as the square roots of the norms
    row_rescale_factor = jnp.sqrt(norm_of_rows)
    column_rescale_factor = jnp.sqrt(norm_of_columns)

    # Scale the problem using these factors
    scale_problem(problem, row_rescale_factor, column_rescale_factor)

    return row_rescale_factor, column_rescale_factor


def rescale_problem(
    l_inf_ruiz_iterations: int,
    l2_norm_rescaling_flag: bool,
    pock_chambolle_alpha: Union[float, None],
    original_problem: QuadraticProgrammingProblem,
) -> ScaledQpProblem:
    """
    Preprocesses and rescales the original problem, returning a ScaledQpProblem struct.

    Parameters
    ----------
    l_inf_ruiz_iterations : int
        The number of iterations for L_inf Ruiz rescaling.
    l2_norm_rescaling_flag : bool
        Whether to apply L2 norm rescaling.
    pock_chambolle_alpha : Union[float, None]
        The exponent parameter for Pock-Chambolle rescaling. Set to None to skip.
    original_problem : QuadraticProgrammingProblem
        The original quadratic programming problem.

    Returns
    -------
    ScaledQpProblem
        A struct containing the scaled problem and rescaling factors.
    """
    # Convert to BCOO format for easier manipulation
    if isinstance(original_problem.constraint_matrix, BCSR):
        original_problem.constraint_matrix = (
            original_problem.constraint_matrix.to_bcoo()
        )
    elif isinstance(original_problem.constraint_matrix, (BCOO, jnp.ndarray)):
        pass
    else:
        raise ValueError("Unsupported matrix format.")

    if isinstance(original_problem.constraint_matrix_t, BCSR):
        original_problem.constraint_matrix_t = (
            original_problem.constraint_matrix_t.to_bcoo()
        )
    elif isinstance(original_problem.constraint_matrix_t, (BCOO, jnp.ndarray)):
        pass
    else:
        raise ValueError("Unsupported matrix format.")

    problem = deepcopy(original_problem)

    num_constraints, num_variables = problem.constraint_matrix.shape
    constraint_rescaling = jnp.ones(num_constraints)
    variable_rescaling = jnp.ones(num_variables)

    if l_inf_ruiz_iterations > 0:
        con_rescale, var_rescale = ruiz_rescaling(
            problem, l_inf_ruiz_iterations, jnp.inf
        )
        constraint_rescaling *= con_rescale
        variable_rescaling *= var_rescale

    if l2_norm_rescaling_flag:
        con_rescale, var_rescale = l2_norm_rescaling(problem)
        constraint_rescaling *= con_rescale
        variable_rescaling *= var_rescale

    if pock_chambolle_alpha is not None:
        con_rescale, var_rescale = pock_chambolle_rescaling(
            problem, pock_chambolle_alpha
        )
        constraint_rescaling *= con_rescale
        variable_rescaling *= var_rescale

    if l_inf_ruiz_iterations == 0 and not l2_norm_rescaling_flag:
        logger.info("No rescaling applied.")
    else:
        logger.info(
            "Problem after rescaling (Ruiz iterations = %d, l2_norm_rescaling = %s):",
            l_inf_ruiz_iterations,
            l2_norm_rescaling_flag,
        )
    display_problem_details(problem)

    if isinstance(original_problem.constraint_matrix, BCOO):
        original_problem.constraint_matrix = BCSR.from_bcoo(
            original_problem.constraint_matrix
        )
    if isinstance(original_problem.constraint_matrix_t, BCOO):
        original_problem.constraint_matrix_t = BCSR.from_bcoo(
            original_problem.constraint_matrix_t
        )
    if isinstance(problem.constraint_matrix, BCOO):
        problem.constraint_matrix = BCSR.from_bcoo(problem.constraint_matrix)
    if isinstance(problem.constraint_matrix_t, BCOO):
        problem.constraint_matrix_t = BCSR.from_bcoo(problem.constraint_matrix_t)
    scaled_problem = ScaledQpProblem(
        original_qp=original_problem,
        scaled_qp=problem,
        constraint_rescaling=constraint_rescaling,
        variable_rescaling=variable_rescaling,
    )

    return scaled_problem


def ruiz_rescaling(
    problem, num_iterations: int, p: float = float("inf")
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Uses a modified Ruiz rescaling algorithm to rescale the matrix M=[Q,A';A,0]
    where Q is objective_matrix and A is constraint_matrix, and returns the
    cumulative scaling vectors.

    Reference:
    https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem. This is modified to store the transformed problem.
    num_iterations : int
        The number of iterations to run the Ruiz rescaling algorithm. Must be positive.
    p : float
        Which norm to use. Must be 2 or Inf.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
        the original problem is recovered by `unscale_problem`.
    """
    num_constraints, num_variables = problem.constraint_matrix.shape
    cum_constraint_rescaling = jnp.ones(num_constraints)
    cum_variable_rescaling = jnp.ones(num_variables)

    for _ in range(num_iterations):
        constraint_matrix = problem.constraint_matrix
        objective_matrix = problem.objective_matrix

        # Determine variable rescaling
        if p == float("inf"):
            constraint_col_max = get_col_l_inf_norms(constraint_matrix)
            objective_col_max = get_col_l_inf_norms(objective_matrix)
            variable_rescaling = jnp.sqrt(
                jnp.maximum(constraint_col_max, objective_col_max)
            )
        elif p == 2:
            variable_rescaling = jnp.sqrt(
                jnp.sqrt(
                    jnp.square(get_col_l2_norms(constraint_matrix))
                    + jnp.square(get_col_l2_norms(objective_matrix))
                )
            )
        else:
            raise ValueError("Norm must be 2 or Inf.")

        # Avoid division by zero by setting zero values to 1.0
        variable_rescaling = jnp.where(
            variable_rescaling == 0.0, 1.0, variable_rescaling
        )

        # Determine constraint rescaling
        if num_constraints == 0:
            constraint_rescaling = jnp.array([])
        else:
            if p == float("inf"):
                constraint_row_max = get_row_l_inf_norms(constraint_matrix)
                constraint_rescaling = jnp.sqrt(constraint_row_max)
            elif p == 2:
                norm_of_rows = get_row_l2_norms(problem.constraint_matrix)

                # Determine the target row norm
                target_row_norm = jnp.sqrt(num_variables / num_constraints)
                if jnp.all(problem.objective_matrix.data == 0):
                    # LP case
                    target_row_norm = jnp.sqrt(num_variables / num_constraints)
                else:
                    # QP case
                    target_row_norm = jnp.sqrt(
                        num_variables / (num_constraints + num_variables)
                    )

                constraint_rescaling = jnp.sqrt(norm_of_rows / target_row_norm)
            else:
                raise ValueError("Norm must be 2 or inf.")

            # Avoid division by zero
            constraint_rescaling = jnp.where(
                constraint_rescaling == 0, 1.0, constraint_rescaling
            )

        # Apply scaling to the problem
        scale_problem(problem, constraint_rescaling, variable_rescaling)

        # Accumulate the cumulative scaling factors
        cum_constraint_rescaling *= constraint_rescaling
        cum_variable_rescaling *= variable_rescaling

    return cum_constraint_rescaling, cum_variable_rescaling


def presolve(
    qp: QuadraticProgrammingProblem, transform_bounds: bool = False
) -> PresolveInfo:
    """
    Preprocess a quadratic program by removing empty rows and columns and
    optionally transforming bounds into constraints.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem to be preprocessed.
    transform_bounds : bool, optional
        Whether to transform finite bounds into linear constraints, by default False.

    Returns
    -------
    PresolveInfo
        A data structure storing information to reverse the presolve transformations.
    """
    saved_variable_lower_bound = jnp.copy(qp.variable_lower_bound)
    saved_variable_upper_bound = jnp.copy(qp.variable_upper_bound)

    original_dual_size, original_primal_size = qp.constraint_matrix.shape

    empty_rows = remove_empty_rows(qp)
    empty_columns = remove_empty_columns(qp)

    check_for_singleton_constraints(qp)

    if transform_bounds:
        transform_bounds_into_linear_constraints(qp)

    return PresolveInfo(
        original_primal_size,
        original_dual_size,
        empty_rows,
        empty_columns,
        saved_variable_lower_bound,
        saved_variable_upper_bound,
    )


def check_for_singleton_constraints(qp: QuadraticProgrammingProblem):
    """
    Identifies and reports constraints that involve only a single variable.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem to be checked.
    """
    num_single = 0
    num_nz_by_row = jnp.zeros(qp.constraint_matrix.shape[0], dtype=jnp.int32)

    for row_ind in qp.constraint_matrix.row_indices:
        num_nz_by_row = num_nz_by_row.at[row_ind].add(1)

    num_single = jnp.sum(num_nz_by_row == 1)

    if num_single > 0:
        logger.info("%d constraints involving exactly a single variable", num_single)


def pock_chambolle_rescaling(
    qp: QuadraticProgrammingProblem, alpha: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies the rescaling proposed by Pock and Cambolle (2011),
    "Diagonal preconditioning for first order primal-dual algorithms
    in convex optimization"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126441&tag=1

    Although presented as a form of diagonal preconditioning, it can be
    equivalently implemented by rescaling the problem data.

    Each column of the constraint matrix is divided by
    sqrt(sum_{elements e in the column} |e|^(2 - alpha))
    and each row of the constraint matrix is divided by
    sqrt(sum_{elements e in the row} |e|^alpha)

    Lemma 2 in Pock and Chambolle demonstrates that this rescaling causes the
    operator norm of the rescaled constraint matrix to be less than or equal to
    one, which is a desirable property for PDHG.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem.
    alpha : float
        Exponent parameter in the range [0, 2].

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        The constraint and variable rescaling factors.
    """
    assert 0 <= alpha <= 2

    constraint_matrix = qp.constraint_matrix
    objective_matrix = qp.objective_matrix
    if isinstance(qp.constraint_matrix, jnp.ndarray):
        variable_rescaling = jnp.sqrt(
            jnp.sum(jnp.abs(constraint_matrix) ** (2 - alpha), axis=0)
            + jnp.sum(jnp.abs(objective_matrix) ** (2 - alpha), axis=0)
        )
        constraint_rescaling = jnp.sqrt(
            jnp.sum(jnp.abs(constraint_matrix) ** (2 - alpha), axis=1)
        )
    elif isinstance(qp.constraint_matrix, BCOO):
        # TODO: improve the code here, instead of using jnp.bincount.
        # Use BCOO.sum or use the sparsify() transform.
        variable_rescaling = jnp.sqrt(
            jnp.bincount(
                constraint_matrix.indices[:, 1],
                weights=jnp.abs(constraint_matrix.data) ** (2 - alpha),
                length=constraint_matrix.shape[1],
            )
            + jnp.bincount(
                objective_matrix.indices[:, 1],
                weights=jnp.abs(objective_matrix.data) ** (2 - alpha),
                length=objective_matrix.shape[1],
            )
        )
        constraint_rescaling = jnp.sqrt(
            jnp.bincount(
                constraint_matrix.indices[:, 0],
                weights=jnp.abs(constraint_matrix.data) ** (alpha),
                length=constraint_matrix.shape[0],
            )
        )

    variable_rescaling = jnp.where(variable_rescaling == 0, 1.0, variable_rescaling)
    constraint_rescaling = jnp.where(
        constraint_rescaling == 0, 1.0, constraint_rescaling
    )

    scale_problem(qp, constraint_rescaling, variable_rescaling)

    return constraint_rescaling, variable_rescaling


def row_permute(matrix: BCOO, old_row_to_new: jnp.ndarray) -> BCOO:
    """
    Permutes the rows of the matrix based on the provided permutation vector and returns a new matrix.

    Parameters
    ----------
    matrix : BCOO
        A sparse matrix in BCOO format (with data and indices).
    old_row_to_new : jnp.ndarray
        A permutation vector that maps old row indices to new row indices.

    Returns
    -------
    BCOO
        A new matrix with permuted rows.
    """
    # Access data and indices directly from the BCOO matrix
    data = matrix.data
    row_indices, col_indices = matrix.indices[:, 0], matrix.indices[:, 1]

    # Apply the row permutation
    new_row_indices = old_row_to_new[row_indices]

    # Sort the row indices within each column
    sorted_indices = jnp.lexsort((col_indices, new_row_indices))

    # Create new BCOO matrix with permuted rows
    new_data = data[sorted_indices]
    new_indices = jnp.column_stack(
        (new_row_indices[sorted_indices], col_indices[sorted_indices])
    )

    # Return a new BCOO matrix
    return BCOO((new_data, new_indices), shape=matrix.shape)
