"""Cross checks for integer map problems
"""
from typing import List, Tuple, Dict, Any

from sympy.core.relational import Relational

from pandas import DataFrame, Series


def check_inequalities(
    data: Tuple[Dict[Any, Any], Series], inequalities: List[Relational]
) -> bool:
    """Checks wether all inequalities are fulfilled by dependents.

    Arguments:
        data:
            Dict or series containing key value pair for all dependents in all
            inequalities. Must contain only those keys or keys which contain
            `constrained`.
        inequalities:
            List of all inequalities.
    """
    if isinstance(data, Series):
        data = data.to_dict()
    dependents = [var for var in data if "constrained" not in str(var)]
    return bool(
        all(
            [ineq.subs({dep: data[dep] for dep in dependents}) for ineq in inequalities]
        )
    )


def run_checks(  # pylint: disable=C0103
    df: DataFrame, inequalities: List[Relational]
) -> bool:
    """Runs the following checks on a constraints table

    1. Checks if all `shifted_value` are larger or equal to zero.
    2. If all inequalities are true for `shifted_value` == 0 parameters.
    3. If any inequality is false for `shifted_value` > 0 parameters.

    Raise:
        AssertionError error if not all checks are fulfilled
    """
    df = df.reset_index()
    if not all(df.shifted_value.unique() >= 0):
        raise AssertionError(
            "Not all values of shifted constrained are larger or equal to zero."
        )

    if not (
        df.query("shifted_value == 0")
        .apply(lambda row: check_inequalities(row, inequalities), axis=1,)
        .astype(bool)
        .all()
    ):
        raise AssertionError(
            "Not all values where the shifted constraints are zero"
            " fulfill all inequalities."
        )

    if (
        df.query("shifted_value > 0")
        .apply(lambda row: check_inequalities(row, inequalities), axis=1,)
        .astype(bool)
        .all()
    ):
        raise AssertionError(
            "Not all values where the shifted constraints are larger than zero"
            " do not fulfill all inequalities."
        )

    return True
