from qcheff.operators.dense_operator import DenseOperator
from qcheff.operators.operator_base import (
    OperatorMatrix,
    qcheff_array,
    qcheff_dense_array,
    qcheff_sparse_array,
)
from qcheff.operators.operators import *  # noqa: F403
from qcheff.operators.sparse_operator import SparseOperator
from qcheff.operators.utils import *  # noqa: F403


def qcheffOperator(op: qcheff_array) -> OperatorMatrix:
    """
    Create an OperatorMatrix object from a given operator.

    Parameters
    ----------
    op : qcheff_array
        The operator to create an OperatorMatrix object from.

    Returns
    -------
    OperatorMatrix
        The created OperatorMatrix object.
    """
    if isinstance(op, qcheff_sparse_array):
        return SparseOperator(op)
    elif isinstance(op, qcheff_dense_array):
        return DenseOperator(op)
    else:
        msg = f"Unsupported operator type: {type(op)}"
        raise TypeError(msg)
