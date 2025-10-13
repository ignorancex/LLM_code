import certifiable_uda_loc.utils.matrix_utils as mat_utils

from typing import List, Union, Tuple, Dict
from poly_matrix import PolyMatrix
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
import numpy as np
import itertools

import certifiable_uda_loc.constraints.constraints_boolean as constraints_boolean
from certifiable_uda_loc.utils.string_utils import (
    var_to_colvar,
    bool_cont_var_name,
)
from certifiable_uda_loc.utils.matrix_utils import np_to_polymatrix
from collections import namedtuple
import certifiable_uda_loc.variables as variables
import certifiable_uda_loc.utils.string_utils as string_utils


def is_both_variables_in_lifted(var1, var2, lifted_variables):
    return var1 in lifted_variables and var2 in lifted_variables


def nonzero(a, tol=1e-7):
    if isinstance(a, np.ndarray):
        a = float(a.squeeze())
    return abs(a) > tol


def x_premultiply_A_theta(
    x: str,
    lifted_variables_names_all: List[str],
    hom_var: variables.HomogenizationVariable,
    bool_lifted_vars: List[variables.LiftedQcQpVariable],
    A_th: PolyMatrix,
):
    """
    Given constraint of form
    [1, \theta^\trans] \mbf{A}_th [1, \\ \theta] =0,
    consider
    x^\trans [1, \theta^\trans] \mbf{A}_th [1, \\ \theta] =0,
    which yields another ndim constraints.
    """
    A_list = []
    nx = len(hom_var.col_names)
    for lv_dim in range(nx):
        A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
        th_vars = [hom_var] + bool_lifted_vars

        for lv1 in range(len(th_vars)):
            for lv2 in range(lv1, len(th_vars)):
                th1_var = th_vars[lv1]
                th2_var = th_vars[lv2]
                if not nonzero(A_th[th1_var.name, th2_var.name]):
                    continue
                A_var_1 = th1_var.column_names()[lv_dim]
                A_var_2 = bool_cont_var_name(th2_var.name, x)

                need_to_scale = (th1_var.name == th2_var.name) and (A_var_1 != A_var_2)
                A[A_var_1, A_var_2] = A_th[th1_var.name, th2_var.name]
                if need_to_scale:
                    A[A_var_1, A_var_2] *= 0.5

        A_list.append(A)

    return A_list


def xi_xj_premultiply_A_theta(
    x1: str,
    x2: str,
    lifted_variables_names_all: List[str],
    bool_variable_names: List[str],
    A_th: PolyMatrix,
):
    """
    Given constraint of form
    [1, \theta^\trans] \mbf{A}_th [1, \\ \theta] =0,
    consider
    x_i^\trans x_j^\trans [1, \theta^\trans] \mbf{A}_th [1, \\ \theta] =0,
    which yields another constraint.
    """
    A_list = []
    valid_constraint = False
    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
    if nonzero(A_th["one", "one"]):
        A[x1, x2] = A_th["one", "one"]
        if x1 != x2:
            A[x1, x2] *= 0.5
    bop = 1
    A[x1, x2] = A_th["one", "one"]
    if x1 != x2:
        A[x1, x2] *= 0.5

    for th1 in bool_variable_names:
        if not nonzero(A_th["one", th1]):
            continue
        A[x1, bool_cont_var_name(th1, x2)] = A_th["one", th1]

    for lv_i, th_i in enumerate(bool_variable_names):
        for lv_j, th_j in enumerate(bool_variable_names):
            if not nonzero(A_th[th_i, th_j]):
                continue
            if lv_j < lv_i:
                continue
            A_var_i = bool_cont_var_name(th_i, x1)
            A_var_j = bool_cont_var_name(th_j, x2)
            A[A_var_i, A_var_j] = A_th[th_i, th_j]
            if (A_var_i != A_var_j) and (th_i == th_j):
                A[bool_cont_var_name(th_i, x1), bool_cont_var_name(th_j, x2)] *= 0.5

    # import certifiable_uda_loc.utils.matrix_utils as mat_utils
    # import certifiable_uda_loc.utils.print_utils as print_utils

    # print("----A_th------")
    # print(A_th)
    # print("-----")
    # mat_utils.print_non_zero_entries(A, symmetric=True)

    A_list.append(A)

    return A_list


def theta_premultiply_A_xi(
    A_xi: PolyMatrix,
    bool_var: variables.LiftedQcQpVariable,
    vector_var_names: List[str],
    hom_var: variables.HomogenizationVariable,
    lifted_variables_names_all: List[str],
) -> List[PolyMatrix]:
    nx = len(hom_var.col_names)
    # First one
    A_list = []
    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
    bool_name = bool_var.name
    for hom_col_name, bool_col_name in zip(hom_var.col_names, bool_var.col_names):
        A[bool_col_name, bool_col_name] = A_xi[hom_col_name, hom_col_name]

    for x_var in vector_var_names:
        bool_cont_name = string_utils.bool_cont_var_name(bool_name, x_var)
        for hom_col_name in hom_var.col_names:
            if not nonzero(A_xi[hom_col_name, x_var]):
                continue
            A[hom_col_name, bool_cont_name] = A_xi[hom_col_name, x_var]
    for x_var1 in vector_var_names:
        for x_var2 in vector_var_names:
            if not nonzero(A_xi[x_var1, x_var2]):
                continue
            bool_cont_name1 = string_utils.bool_cont_var_name(bool_name, x_var1)
            bool_cont_name2 = string_utils.bool_cont_var_name(bool_name, x_var2)
            A[bool_cont_name1, bool_cont_name2] = A_xi[x_var1, x_var2]
    A_list.append(A)

    if np.linalg.norm(A.get_matrix_sparse().todense(), "fro") < 1e-10:
        bop = 1
    # Second one
    # TODO: DOUBLE CHECK. THIS HAS A WEIRD INTERACTION WITH ENFORCE FIRST POSE CONSTRAINT... FIX???
    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
    bool_name = bool_var.name
    for hom_col_name, bool_col_name in zip(hom_var.col_names, bool_var.col_names):
        A[bool_col_name, hom_col_name] = 0.5 * A_xi[hom_col_name, hom_col_name]
    for x_var in vector_var_names:
        bool_cont_name = string_utils.bool_cont_var_name(bool_name, x_var)
        for lv_dim in range(nx):
            if not nonzero(A_xi[x_var, hom_var.col_names[lv_dim]]):
                continue
            A[bool_cont_name, hom_var.col_names[lv_dim]] = A_xi[
                x_var, hom_var.col_names[lv_dim]
            ]
    for x_var1 in vector_var_names:
        for x_var2 in vector_var_names:
            if not nonzero(A_xi[x_var1, x_var2]):
                continue
            bool_cont_name1 = string_utils.bool_cont_var_name(bool_name, x_var1)
            A[bool_cont_name1, x_var2] = A_xi[x_var1, x_var2]
            if x_var1 == x_var2:
                A[bool_cont_name1, x_var2] = A[bool_cont_name1, x_var2] * 0.5
    A_list.append(A)

    return A_list


def theta_premultiply_A_xi_with_theta_i_theta_j(
    A_xi: PolyMatrix,
    bool_var1: variables.LiftedQcQpVariable,
    bool_var2: variables.LiftedQcQpVariable,
    vector_var_names: List[str],
    hom_var: variables.HomogenizationVariable,
    lifted_variables_names_all: List[str],
    version2=False,
) -> List[PolyMatrix]:
    assert bool_var1.name != bool_var2.name
    nx = len(hom_var.col_names)
    # First one
    A_list = []
    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)

    for hom_col_name, bool1_col_name, bool2_col_name in zip(
        hom_var.col_names, bool_var1.col_names, bool_var2.col_names
    ):
        A[bool1_col_name, bool2_col_name] = A_xi[hom_col_name, hom_col_name]
        if bool1_col_name != bool2_col_name:
            A[bool1_col_name, bool2_col_name] *= 0.5

    for x_var in vector_var_names:
        bool_cont_name1 = string_utils.bool_cont_var_name(bool_var1.name, x_var)
        bool_cont_name2 = string_utils.bool_cont_var_name(bool_var2.name, x_var)

        for hom_col_name, bool2_col_name in zip(hom_var.col_names, bool_var2.col_names):
            if not nonzero(A_xi[hom_col_name, x_var]):
                continue
            A[bool_cont_name1, bool2_col_name] = A_xi[hom_col_name, x_var]

    for x_var1, x_var2 in itertools.combinations_with_replacement(vector_var_names, 2):
        if not version2:
            bool_cont_name1 = string_utils.bool_cont_var_name(bool_var1.name, x_var1)
            bool_cont_name2 = string_utils.bool_cont_var_name(bool_var2.name, x_var2)
        if version2:
            bool_cont_name1 = string_utils.bool_cont_var_name(bool_var2.name, x_var1)
            bool_cont_name2 = string_utils.bool_cont_var_name(bool_var1.name, x_var2)
        if not nonzero(A_xi[x_var1, x_var2]):
            continue
        A[bool_cont_name1, bool_cont_name2] = A_xi[x_var1, x_var2]
        if x_var1 == x_var2:
            A[bool_cont_name1, bool_cont_name2] *= 0.5
    A_list.append(A)
    return A_list


# def theta_premultiply_A_xi_with_theta_i_theta_j_version2(
#     A_xi: PolyMatrix,
#     bool_var1: variables.LiftedQcQpVariable,
#     bool_var2: variables.LiftedQcQpVariable,
#     vector_var_names: List[str],
#     hom_var: variables.HomogenizationVariable,
#     lifted_variables_names_all: List[str],
# ) -> List[PolyMatrix]:
#     # Second one.. only the last part is different.
#     A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)

#     for hom_col_name, bool1_col_name, bool2_col_name in zip(
#         hom_var.col_names, bool_var1.col_names, bool_var2.col_names
#     ):
#         A[bool1_col_name, bool2_col_name] = A_xi[hom_col_name, hom_col_name]

#     for x_var in vector_var_names:
#         bool_cont_name1 = string_utils.bool_cont_var_name(bool_var1.name, x_var)
#         bool_cont_name2 = string_utils.bool_cont_var_name(bool_var2.name, x_var)

#         for hom_col_name, bool2_col_name in zip(hom_var.col_names, bool_var2.col_names):
#             A[bool_cont_name1, bool2_col_name] = A_xi[hom_col_name, x_var]

#     for x_var1, x_var2 in itertools.combinations_with_replacement(vector_var_names, 2):
#         bool_cont_name1 = string_utils.bool_cont_var_name(bool_var2.name, x_var1)
#         bool_cont_name2 = string_utils.bool_cont_var_name(bool_var1.name, x_var2)
#         A[bool_cont_name1, bool_cont_name2] = A_xi[x_var1, x_var2]
#     A_list.append(A)

#     # print("---A_xi")
#     # mat_utils.print_non_zero_entries(A_xi)
#     # print("----Resultant A list 0")
#     # mat_utils.print_non_zero_entries(A_list[0])
#     # bop = 1
#     return A_list


def premultiply_sum_constraint(
    hom_var: str,
    A_in: PolyMatrix,
    bool_var_names_to_premult: List[str],
    bool_var_premult: str,
    lifted_variable_names: List[str],
) -> List[PolyMatrix]:
    """
    Premultiply linear constraint on discrete variables by another discrete variable.
    This operates in 1d.. in that lifted_variable_names do not have different columns.
    Its just [bool1 bool2 bool3 etc]
    """
    A: PolyMatrix = initialize_poly_matrix(lifted_variable_names)
    A[hom_var, bool_var_premult] = A_in[hom_var, hom_var] / 2
    for b1 in bool_var_names_to_premult:
        A[b1, bool_var_premult] = A_in[hom_var, b1]
    return [A]


# Interfactor constraint.
def boolean_premultiply_constraint(
    hom_var: variables.HomogenizationVariable,
    bool_names_to_premultiply: List[str],
    bool_var_premultiplying: str,
    b_hom: float,
    b: np.ndarray,
    lifted_variables_names_all: List[str],
) -> PolyMatrix:
    lv_dim = 0
    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)

    # Set the one entry
    A_var_2 = var_to_colvar(bool_var_premultiplying, 0)
    A[hom_var.column_names()[0], A_var_2] = 0.5 * b_hom
    # Set the non-one entries
    A_var_2 = var_to_colvar(bool_var_premultiplying, 0)
    for lv1, th1 in enumerate(bool_names_to_premultiply):
        A_var_1 = var_to_colvar(th1, lv_dim)
        if A_var_1 == A_var_2:
            A[A_var_1, A_var_2] = b[lv1]
        if A_var_1 != A_var_2:
            A[A_var_1, A_var_2] = 0.5 * b[lv1]
    return A


def boolean_premultiply_with_x(
    hom_var: variables.HomogenizationVariable,
    bool_names_to_premultiply: List[str],
    bool_var_premultiplying: str,
    b_hom: float,
    b: np.ndarray,
    lifted_variables_names_all: List[str],
    x_var: str,
):
    nx = hom_var.dims[0]
    A_list = []
    # for lv_dim in range(nx):
    lv_dim = 0
    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
    lv_bool_names_to_premultiply = [
        var_to_colvar(name, lv_dim) for name in bool_names_to_premultiply
    ]
    lv_hom = hom_var.column_names()[lv_dim]
    A_var2 = bool_cont_var_name(bool_var_premultiplying, x_var)
    if nonzero(b_hom):
        A[lv_hom, A_var2] = b_hom

    for lv1 in range(len(bool_names_to_premultiply)):
        A_var1 = lv_bool_names_to_premultiply[lv1]
        if nonzero(b[lv1]):
            A[A_var1, A_var2] = b[lv1]
    A_list.append(A)
    # lv_bool_var_premultiplying = var_to_colvar(bool_var_premultiplying, lv_dim)

    return A_list
