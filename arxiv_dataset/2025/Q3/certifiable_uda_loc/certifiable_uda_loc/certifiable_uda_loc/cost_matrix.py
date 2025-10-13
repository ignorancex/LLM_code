import certifiable_uda_loc.lifting as lifting
from certifiable_uda_loc.lifting import QuadraticForm
from certifiable_uda_loc.variables import ColumnQcQcpVariable, HomogenizationVariable
import itertools
from typing import List
import numpy as np
from certifiable_uda_loc.variables import LiftedQcQpVariable
import certifiable_uda_loc.problem as rf
from py_factor_graph.data_associations import (
    VectorVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
    PRIOR_FACTOR_TYPES,
    PRIOR_FACTOR_TYPES_LIST,
)
import py_factor_graph.data_associations as pyfg_da

import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from typing import List, Tuple
from poly_matrix import PolyMatrix
from certifiable_uda_loc.utils.string_utils import var_to_colvar
import certifiable_uda_loc.utils.print_utils as print_utils
import certifiable_uda_loc.utils.matrix_utils as mat_utils
from typing import Union
import certifiable_uda_loc.problem_setup as problem_setup
import certifiable_uda_loc.variables as variables
import certifiable_uda_loc.lifting as lifting

import certifiable_uda_loc.utils.string_utils as string_utils
import certifiable_uda_loc.lifting as lifting


# The structure of this whole file is kinda cancer


def Q_multiplied_by_boolean(
    hom_var: HomogenizationVariable,
    bool_var: LiftedQcQpVariable,
    cont_var_colnames: List[str],
    Q_orig: PolyMatrix,
) -> PolyMatrix:
    # To go from a cost matrix defined for
    # <Q_orig, X^\trans X> where X = [\eye x]
    # to
    # \theta <Q, X^\trans X>
    #  <Q_new, \thetaX^\trans X>
    # <Q_new, X_new^\trans X_new>
    # with X_new having the boolean columns. X_new = [\theta (\theta X)].

    # TODO: I need to figure out the cross terms. They are missing lol.
    bool_cont_var_names = [
        string_utils.bool_cont_var_name(bool_var.name, cont_var_colname)
        for cont_var_colname in cont_var_colnames
    ]

    lifted_variables = (
        hom_var.column_names() + bool_var.column_names() + bool_cont_var_names
    )
    Q_uda_comp: PolyMatrix = initialize_poly_matrix(lifted_variables)

    for hom_colvar, th_colvar in zip(hom_var.column_names(), bool_var.column_names()):
        Q_uda_comp[th_colvar, th_colvar] = Q_orig[hom_colvar, hom_colvar]

    for bool_cont_var_name, cont_var_colname in zip(
        bool_cont_var_names, cont_var_colnames
    ):
        Q_uda_comp[bool_cont_var_name, bool_cont_var_name] = Q_orig[
            cont_var_colname, cont_var_colname
        ]

        for hom_colvar, th_colvar in zip(
            hom_var.column_names(), bool_var.column_names()
        ):
            Q_uda_comp[th_colvar, bool_cont_var_name] = Q_orig[
                hom_colvar, cont_var_colname
            ]

    for cont_var1, cont_var2 in itertools.combinations(cont_var_colnames, 2):
        if cont_var1 == cont_var2:
            continue
        bool_cont_var_name1 = string_utils.bool_cont_var_name(bool_var.name, cont_var1)
        Q_uda_comp[bool_cont_var_name1, cont_var2] = Q_orig[cont_var1, cont_var2]

    return Q_uda_comp


def get_cost_matrix(
    meas_list: List[Union[UnknownDataAssociationMeasurement, ToyExamplePrior]],
    column_lifted_variables: List[ColumnQcQcpVariable],
    safe_check: bool = False,
    verbose=False,
    sparse_bool_cont_variables=False,
) -> PolyMatrix:
    nx = column_lifted_variables[0].dims[0]
    boolean_vars_all = []
    lifted_variables_names_all = []

    uda_meas_list = []
    standard_meas_list = []
    for meas in meas_list:
        if isinstance(meas, UnknownDataAssociationMeasurement):
            uda_meas_list.append(meas)
        else:
            standard_meas_list.append(meas)

    for uda_meas in uda_meas_list:
        boolean_vars_all += uda_meas.boolean_variables

    hom_var = variables.HomogenizationVariable((nx, nx), "one", true_value=np.eye(nx))

    _, lifted_variables_names_all = lifting.get_lifted_variables_all(
        hom_var,
        column_lifted_variables,
        uda_meas_list,
        sparse_bool_cont_variables=sparse_bool_cont_variables,
    )

    # TODO: Streamline this, theres a lot garbage copy paste here..
    Q_all = rf.initialize_poly_matrix(lifted_variables_names_all)

    Q_standard_list_of_lists: List[List[PolyMatrix]] = []
    for meas in standard_meas_list:

        Q_list = lifting.lift_factor_to_Q(meas, hom_var)

        Q_standard_list_of_lists.append(Q_list)

        for Q in Q_list:
            Q_all = Q_all + Q

    if np.any(np.isinf(Q_all.toarray())):
        raise (BaseException("Infinite entries in the cost matrix..."))
    # Q_uda = rf.initialize_poly_matrix(lifted_variables_names_all)
    for uda_meas in uda_meas_list:
        Q_comp_list = []
        for bool_var, meas in zip(
            uda_meas.boolean_variables, uda_meas.measurement_list
        ):
            # Does homogenization variable get added multiple times?
            # Doing this for every column seems iffy.
            Q_std_list = lifting.lift_factor_to_Q(meas, hom_var)
            _, col_names = lifting.get_column_names_from_factor(meas)
            bool_lifted_var = lifting.lift_scalar_variable(bool_var, nx)
            Q_booled_list = []
            for Q_std in Q_std_list:
                Q_booled = Q_multiplied_by_boolean(
                    hom_var,
                    bool_lifted_var,
                    col_names,
                    Q_std,
                )
                Q_booled_list.append(Q_booled)

                Q_all = Q_all + Q_booled

                # Q_uda = Q_uda + Q_booled

    return Q_all
