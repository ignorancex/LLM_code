import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR, bcoo_concatenate, empty
from scipy.sparse import sparray, spmatrix

from mpax.utils import QuadraticProgrammingProblem, TwoSidedQpProblem


def transform_to_standard_form(qp: TwoSidedQpProblem) -> QuadraticProgrammingProblem:
    """
    Transforms a quadratic program into a `QuadraticProgrammingProblem` named tuple.
    The `TwoSidedQpProblem` is destructively modified in place to avoid creating a copy.

    Parameters
    ----------
    qp : TwoSidedQpProblem
        The quadratic programming problem to transform.

    Returns
    -------
    QuadraticProgrammingProblem
        A named tuple containing the transformed quadratic programming problem.
    """
    two_sided_rows_to_slacks(qp)

    is_equality_row = jnp.equal(qp.constraint_lower_bound, qp.constraint_upper_bound)
    is_geq_row = jnp.logical_not(is_equality_row) & jnp.isfinite(
        qp.constraint_lower_bound
    )
    is_leq_row = jnp.logical_not(is_equality_row) & jnp.isfinite(
        qp.constraint_upper_bound
    )

    assert not jnp.any(
        is_geq_row & is_leq_row
    ), "Two-sided rows should be removed by two_sided_rows_to_slacks."

    num_equalities = jnp.sum(is_equality_row)

    if num_equalities + jnp.sum(is_geq_row) + jnp.sum(is_leq_row) != len(
        qp.constraint_lower_bound
    ):
        raise ValueError("Not all constraints have finite bounds on at least one side.")

    # Extract row indices from BCOO format
    row_indices = qp.constraint_matrix.indices[
        :, 0
    ]  # First column of indices is the row indices

    # Flip the signs of the leq rows in place
    leq_nzval_mask = jnp.take(is_leq_row, row_indices)
    constraint_matrix_data = jnp.where(
        leq_nzval_mask, -qp.constraint_matrix.data, qp.constraint_matrix.data
    )

    constraint_matrix = BCOO(
        (constraint_matrix_data, qp.constraint_matrix.indices),
        shape=qp.constraint_matrix.shape,
    )
    constraint_matrix_t = BCSR.from_bcoo(constraint_matrix.T)

    # Update the right-hand-side for <= rows, then apply the same row permutation
    right_hand_side = jnp.where(
        is_leq_row, -qp.constraint_upper_bound, qp.constraint_lower_bound
    )
    num_constraints, num_variables = qp.constraint_matrix.shape
    isfinite_variable_lower_bound = jnp.isfinite(qp.variable_lower_bound)
    isfinite_variable_upper_bound = jnp.isfinite(qp.variable_upper_bound)

    equalities_mask = is_equality_row
    inequalities_mask = ~is_equality_row

    return QuadraticProgrammingProblem(
        num_variables=num_variables,
        num_constraints=num_constraints,
        variable_lower_bound=qp.variable_lower_bound,
        variable_upper_bound=qp.variable_upper_bound,
        isfinite_variable_lower_bound=isfinite_variable_lower_bound,
        isfinite_variable_upper_bound=isfinite_variable_upper_bound,
        objective_matrix=qp.objective_matrix,
        objective_vector=qp.objective_vector,
        objective_constant=qp.objective_constant,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix_t,
        right_hand_side=right_hand_side,
        num_equalities=num_equalities,
        equalities_mask=equalities_mask,
        inequalities_mask=inequalities_mask,
    )


def two_sided_rows_to_slacks(qp: TwoSidedQpProblem) -> None:
    """
    Transforms a `TwoSidedQpProblem` in-place to remove two-sided constraints, i.e.,
    constraints with lower and upper bounds that are both finite and not equal.

    For each two-sided constraint, a slack variable is added, and the constraint is changed
    to an equality: `l <= a'x <= u` becomes `a'x - s = 0, l <= s <= u`.

    Parameters
    ----------
    qp : TwoSidedQpProblem
        The quadratic programming problem with two-sided constraints.

    Returns
    -------
    None
        The function modifies the input `qp` in-place to handle two-sided constraints.
    """

    # Find all rows where the constraints are two-sided and bounds are not equal
    two_sided_rows = jnp.where(
        jnp.isfinite(qp.constraint_lower_bound)
        & jnp.isfinite(qp.constraint_upper_bound)
        & (qp.constraint_lower_bound != qp.constraint_upper_bound)
    )[0]

    if len(two_sided_rows) == 0:
        return

    # Construct a slack matrix using the indices of two-sided constraints
    row_indices = two_sided_rows
    col_indices = jnp.arange(len(two_sided_rows))
    slack_values = -jnp.ones(len(two_sided_rows))

    # Create BCOO slack matrix
    slack_matrix = BCOO(
        (slack_values, jnp.stack((row_indices, col_indices), axis=-1)),
        shape=(len(qp.constraint_lower_bound), len(two_sided_rows)),
    )

    # Update variable bounds by adding slack variable bounds
    qp.variable_lower_bound = jnp.concatenate(
        [qp.variable_lower_bound, qp.constraint_lower_bound[two_sided_rows]]
    )
    qp.variable_upper_bound = jnp.concatenate(
        [qp.variable_upper_bound, qp.constraint_upper_bound[two_sided_rows]]
    )

    # Update the objective vector by adding zeros for the slack variables
    qp.objective_vector = jnp.concatenate(
        [qp.objective_vector, jnp.zeros(len(two_sided_rows))]
    )

    # Append the slack matrix to the constraint matrix
    qp.constraint_matrix = bcoo_concatenate(
        [qp.constraint_matrix, slack_matrix], dimension=1
    )

    # Set the bounds of the two-sided constraints to 0
    qp.constraint_lower_bound = qp.constraint_lower_bound.at[two_sided_rows].set(0)
    qp.constraint_upper_bound = qp.constraint_upper_bound.at[two_sided_rows].set(0)

    # Update the objective matrix to accommodate the new variables
    new_num_variables = len(qp.variable_lower_bound)
    row_indices, col_indices, nonzeros = (
        qp.objective_matrix.indices[:, 0],
        qp.objective_matrix.indices[:, 1],
        qp.objective_matrix.data,
    )
    qp.objective_matrix = BCOO(
        (nonzeros, jnp.stack((row_indices, col_indices), axis=-1)),
        shape=(new_num_variables, new_num_variables),
    )


def transform_to_bcoo(input_matrix):
    """Transform the input matrix to the BCOO format.

    Parameters
    ----------
    input_matrix :
        The input matrix to be transformed.

    Returns
    -------
    BCOO
        The matrix in BCOO format.

    Raises
    ------
    ValueError
        The input matrix format is not supported.
    """
    if isinstance(input_matrix, (jnp.ndarray, np.ndarray)):
        bcoo_matrix = BCOO.fromdense(
            input_matrix, nse=input_matrix.shape[0] * input_matrix.shape[1]
        )
    elif isinstance(input_matrix, BCSR):
        bcoo_matrix = input_matrix.to_bcoo()
    elif isinstance(input_matrix, (sparray, spmatrix)):
        bcoo_matrix = BCOO.from_scipy_sparse(input_matrix)
    elif isinstance(input_matrix, BCOO):
        bcoo_matrix = input_matrix
    else:
        raise ValueError(
            "Unsupported matrix format. "
            "The constraint matrix must be one of the following types: "
            "jnp.ndarray, numpy.ndarray, BCOO, or BCSR."
        )
    return bcoo_matrix


def transform_to_jnp_array(input_matrix):
    """Transform the input matrix to the BCOO format.

    Parameters
    ----------
    input_matrix :
        The input matrix to be transformed.

    Returns
    -------
    jnp.ndarray
        The matrix in jnp.ndarray format.

    Raises
    ------
    ValueError
        The input matrix format is not supported.
    """
    if isinstance(input_matrix, (BCOO, BCSR)):
        output_matrix = input_matrix.todense()
    elif isinstance(input_matrix, np.ndarray):
        output_matrix = jnp.array(input_matrix)
    elif isinstance(input_matrix, (sparray, spmatrix)):
        output_matrix = jnp.array(input_matrix.toarray())
    elif isinstance(input_matrix, jnp.ndarray):
        output_matrix = input_matrix
    else:
        raise ValueError(
            "Unsupported matrix format. "
            "The constraint matrix must be one of the following types: "
            "jnp.ndarray, BCOO, or BCSR."
        )
    return output_matrix


def create_lp(c, A, b, G, h, l, u, use_sparse_matrix=True):
    """Create a boxed linear program from arrays.
            max  cx
            s.t. Ax = b
                 Gx >= h
                 l <= x <= u

    Parameters
    ----------
    c : jnp.ndarray
        The objective vector.
    A : jnp.ndarray, BCOO or BCSR
        The matrix of equality constraints.
    b : jnp.ndarray
        The right hand side of equality constraints.
    G : jnp.ndarray, BCOO or BCSR
        The matrix for inequality constraints.
    b : jnp.ndarray
        The right hand side of inequality constraints.
    l : jnp.ndarray
        The lower bound of the variables.
    u : jnp.ndarray
        The upper bound of the variables.
    use_sparse_matrix: bool
        Whether to use sparse matrix format, by default True.

    Returns
    -------
    QuadraticProgrammingProblem
        The boxed linear program.
    """
    if use_sparse_matrix:
        constraint_matrix = bcoo_concatenate(
            [transform_to_bcoo(A), transform_to_bcoo(G)], dimension=0
        )
        objective_matrix = empty(
            shape=(len(c), len(c)), dtype=float, sparse_format="bcoo"
        )
    else:
        constraint_matrix = jnp.concatenate(
            [transform_to_jnp_array(A), transform_to_jnp_array(G)], axis=0
        )
        objective_matrix = jnp.empty(shape=(len(c), len(c)), dtype=float)

    problem = QuadraticProgrammingProblem(
        num_variables=c.shape[0],
        num_constraints=A.shape[0] + G.shape[0],
        variable_lower_bound=jnp.array(l),
        variable_upper_bound=jnp.array(u),
        isfinite_variable_lower_bound=jnp.isfinite(l),
        isfinite_variable_upper_bound=jnp.isfinite(u),
        objective_matrix=objective_matrix,
        objective_vector=jnp.array(c),
        objective_constant=0.0,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix.T,
        right_hand_side=jnp.concatenate([b, h]),
        num_equalities=b.shape[0],
        equalities_mask=jnp.concatenate(
            [jnp.full(len(b), True), jnp.full(len(h), False)]
        ),
        inequalities_mask=jnp.concatenate(
            [jnp.full(len(b), False), jnp.full(len(h), True)]
        ),
        is_lp=True,
    )
    return problem


def create_qp(Q, c, A, b, G, h, l, u, use_sparse_matrix=True):
    """Create a boxed linear program from arrays.
            max  cx
            s.t. Ax = b
                 Gx >= h
                 l <= x <= u

    Parameters
    ----------
    Q : jnp.ndarray, BCOO or BCSR
        The matrix of the quadratic term.
    c : jnp.ndarray
        The objective vector.
    A : jnp.ndarray, BCOO or BCSR
        The matrix of equality constraints.
    b : jnp.ndarray
        The right hand side of equality constraints.
    G : jnp.ndarray, BCOO or BCSR
        The matrix for inequality constraints.
    b : jnp.ndarray
        The right hand side of inequality constraints.
    l : jnp.ndarray
        The lower bound of the variables.
    u : jnp.ndarray
        The upper bound of the variables.
    use_sparse_matrix: bool
        Whether to use sparse matrix format, by default True.

    Returns
    -------
    QuadraticProgrammingProblem
        The boxed linear program.
    """
    if use_sparse_matrix:
        constraint_matrix = bcoo_concatenate(
            [transform_to_bcoo(A), transform_to_bcoo(G)], dimension=0
        )
        objective_matrix = transform_to_bcoo(Q)
        is_lp = objective_matrix.nse == 0
    else:
        constraint_matrix = jnp.concatenate(
            [transform_to_jnp_array(A), transform_to_jnp_array(G)], axis=0
        )
        objective_matrix = transform_to_jnp_array(Q)
        is_lp = jnp.all(objective_matrix == 0)

    problem = QuadraticProgrammingProblem(
        num_variables=c.shape[0],
        num_constraints=A.shape[0] + G.shape[0],
        variable_lower_bound=jnp.array(l),
        variable_upper_bound=jnp.array(u),
        isfinite_variable_lower_bound=jnp.isfinite(l),
        isfinite_variable_upper_bound=jnp.isfinite(u),
        objective_matrix=objective_matrix,
        objective_vector=jnp.array(c),
        objective_constant=0.0,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix.T,
        right_hand_side=jnp.concatenate([b, h]),
        num_equalities=b.shape[0],
        equalities_mask=jnp.concatenate(
            [jnp.full(len(b), True), jnp.full(len(h), False)]
        ),
        inequalities_mask=jnp.concatenate(
            [jnp.full(len(b), False), jnp.full(len(h), True)]
        ),
        is_lp=is_lp,
    )
    return problem


def create_qp_from_gurobi(
    model, use_sparse_matrix=True, sharding=None
) -> QuadraticProgrammingProblem:
    """Transforms a gurobi model to a standard form.

    Parameters
    ----------
    model : gurobipy.Model
        The gurobi model to transform.
    use_sparse_matrix : bool
        Whether to use sparse matrix format, by default True.
    sharding : jax.sharding.Sharding
        The sharding to use, by default None.

    Returns
    -------
    QuadraticProgrammingProblem
        The standard form of the problem.
    """
    constraint_sense = np.array(model.getAttr("Sense", model.getConstrs()))
    leq_mask = jnp.array(constraint_sense == "<")
    equalities_mask = jnp.array(constraint_sense == "=")

    # Flip the signs of the leq rows in place
    constraint_rhs = jnp.array(model.getAttr("RHS", model.getConstrs()))
    constraint_rhs = jnp.where(leq_mask, -constraint_rhs, constraint_rhs)

    if use_sparse_matrix:
        constraint_matrix = BCOO.from_scipy_sparse(model.getA())
        row_indices = constraint_matrix.indices[:, 0]
        leq_nzval_mask = jnp.take(jnp.array(leq_mask), row_indices)
        constraint_matrix.data = jnp.where(
            leq_nzval_mask, -constraint_matrix.data, constraint_matrix.data
        )
        objective_matrix = 2 * BCOO.from_scipy_sparse(
            (model.getQ() + model.getQ().T) / 2
        )
    else:
        constraint_matrix = jnp.array(model.getA().toarray())
        constraint_matrix = constraint_matrix.at[leq_mask].set(
            -constraint_matrix[leq_mask]
        )
        objective_matrix = 2 * jnp.array(
            ((model.getQ() + model.getQ().T) / 2).toarray()
        )
    is_lp = model.getQ().nnz == 0
    if sharding is not None:
        constraint_matrix = jax.device_put(constraint_matrix, sharding)

    var_lb = jnp.array(model.getAttr("LB", model.getVars()))
    var_ub = jnp.array(model.getAttr("UB", model.getVars()))

    objective_vector = jnp.array(model.getAttr("Obj", model.getVars()))
    objective_constant = model.ObjCon

    problem = QuadraticProgrammingProblem(
        num_variables=len(var_lb),
        num_constraints=len(constraint_rhs),
        variable_lower_bound=var_lb,
        variable_upper_bound=var_ub,
        isfinite_variable_lower_bound=jnp.isfinite(var_lb),
        isfinite_variable_upper_bound=jnp.isfinite(var_ub),
        objective_matrix=objective_matrix,
        objective_vector=objective_vector,
        objective_constant=objective_constant,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix.T,
        right_hand_side=constraint_rhs,
        num_equalities=jnp.sum(equalities_mask),
        equalities_mask=equalities_mask,
        inequalities_mask=~equalities_mask,
        is_lp=is_lp,
    )
    return problem
