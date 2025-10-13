import numpy as np
import certifiable_uda_loc.problem as rf

import numpy as np
import certifiable_uda_loc.utils.matrix_utils as mat_utils
import cert_tools

import cvxpy as cp
from certifiable_uda_loc.problem import is_col_var

from certifiable_uda_loc.variables import LiftedQcQpVariable
from certifiable_uda_loc.utils import print_utils
import cvxpy as cp
from py_factor_graph.data_associations import UnknownDataAssociationMeasurement
from typing import List
from tqdm import tqdm
from poly_matrix import PolyMatrix
from certifiable_uda_loc.redundant_constraints import construct_redundant as constr_red
import itertools
from collections import namedtuple

PolyMatEntry = namedtuple("PolyMatEntry", "var1 var2 val")


def find_sparse_constraints(
    feasible_point_mat_list: List[PolyMatrix],
    existing_constraint_mat_list: List[np.ndarray],
    lifted_variables_names_all: List[str],
    formulation="orthogonal",
    verbose=True,
):
    vechX_list = [
        mat_utils.mat2vech(x.toarray().T @ x.toarray()) for x in feasible_point_mat_list
    ]

    VechX = np.hstack([v.reshape(-1, 1) for v in vechX_list])
    # print_utils.pretty_print_array(
    #     VechX,
    #     [""] * VechX.shape[0],
    #     [""] * VechX.shape[1],
    # )
    possible_num_constraints = VechX.shape[0] - np.linalg.matrix_rank(VechX)
    if verbose:
        print("Dim of free variables in lifted matrix", VechX.shape[0])
        print("Number of cols in VechX", VechX.shape[1])
        print("Rank of VechX", np.linalg.matrix_rank(VechX))
        print("Possible number of constraints", possible_num_constraints)
    m_vec_list = []
    n_constr_min_card = []
    VechA = None
    # possible_num_constraints = 5
    if formulation != "qrp":
        for lv1 in tqdm(range(possible_num_constraints)):
            if m_vec_list:
                if m_vec_list[-1] is None:
                    m_vec_list = m_vec_list[:-1]
                    break
                VechA = np.hstack([v.reshape(-1, 1) for v in m_vec_list])
            m_list_lv1 = find_possible_constraint_choices(
                VechX, VechA, formulation=formulation, verbose=verbose
            )
            card_list = [cardinality(m) for m in m_list_lv1]
            for lv1 in range(len(card_list)):
                if card_list[lv1] is None:
                    card_list[lv1] = 1e10

            min_card = np.min(card_list)
            indices = [
                lv1 for lv1 in range(len(card_list)) if card_list[lv1] == min_card
            ]
            idx = indices[0]
            n_constraints_with_min_card = len(indices)
            m_found = m_list_lv1[idx]
            # cap_A = np.hstack([cap_A, m_found.reshape(-1, 1)])
            m_vec_list.append(m_found)
            n_constr_min_card.append(n_constraints_with_min_card)

    if formulation == "qrp":
        nullspace, info = cert_tools.linalg_tools.get_nullspace(VechX.T, tolerance=1e-8)
        nullspace = nullspace.T
        m_vec_list = [nullspace[:, lv1] for lv1 in range(nullspace.shape[1])]

    for lv1 in range(len(m_vec_list)):
        m_vec = m_vec_list[lv1]
        M = mat_utils.vech2mat(m_vec, N=len(lifted_variables_names_all))
        # print_utils.pretty_print_array(
        #     M,
        #     lifted_variables_names_all,
        #     lifted_variables_names_all,
        # )
        # Have to figure out a way to print these.
        M_poly = mat_utils.np_to_polymatrix(
            lifted_variables_names_all, lifted_variables_names_all, M
        )
        # print(f"M at iteration {lv1}")
        # for var1 in lifted_variables_names_all:
        #     for var2 in lifted_variables_names_all:
        #         if np.abs(M_poly[var1, var2]) > 1e-2:
        #             print(f"{var1}, {var2} = {M_poly[var1, var2].squeeze()}")

    return m_vec_list


def cardinality(m: np.ndarray, threshold: float = 1e-2) -> int:
    if m is None:
        return None
    sparsity_vec = np.abs(m) > threshold
    card = len([b for b in sparsity_vec if b])
    return card


def find_possible_constraint_choices(
    VechX, VechA=None, formulation="orthogonal", verbose=True
):
    m_list = []
    m = cp.Variable((VechX.shape[0]))

    if formulation == "v_sum_one":
        m = cp.Variable((VechX.shape[0]))
        v = cp.Variable((VechX.shape[0]))
        b = cp.Variable((VechA.shape[1]))
        constraints = (
            [VechX.T @ m == np.zeros(VechX.shape[1])]
            + [v[lv_search] == 1]
            + [VechA @ b + v == m]
        )
        prob = cp.Problem(cp.Minimize(cp.norm(m, p=1)), constraints=constraints)

    if formulation == "orthogonal" or formulation == "non-orthogonal":
        for lv_search in range(VechX.shape[0]):
            if formulation == "orthogonal" or VechA is None:
                capA = VechX
                if VechA is not None:
                    capA = np.hstack([VechA, VechX])

                constraints = [capA.T @ m == np.zeros(capA.shape[1])] + [
                    m[lv_search] == 1
                ]
                prob = cp.Problem(cp.Minimize(cp.norm(m, p=1)), constraints=constraints)

            if formulation == "non-orthogonal" and VechA is not None:
                m = cp.Variable((VechX.shape[0]))
                v = cp.Variable((VechX.shape[0]))
                b = cp.Variable((VechA.shape[1]))
                constraints = (
                    [VechX.T @ m == np.zeros(VechX.shape[1])]
                    + [v[lv_search] == 1]
                    + [VechA @ b + v == m]
                )
                prob = cp.Problem(cp.Minimize(cp.norm(m, p=1)), constraints=constraints)
            try:
                prob.solve(verbose=verbose)
            except:
                m_list.append(None)
            if prob.value and np.isfinite(prob.value):
                m_vec = m.value
                if np.linalg.norm(m_vec) < 1e-3:
                    m_vec = None
                m_list.append(m_vec)
            else:
                m_list.append(None)

    return m_list


def get_feasible_points(
    uda_meas_list,
    opt_variables: List[LiftedQcQpVariable],
    lifted_variables_names_all,
    # locked_variable_indices: int = [],
    n_random_x_points_per_bool_combo=10,
) -> List[PolyMatrix]:

    uda_meas_list: List[UnknownDataAssociationMeasurement] = uda_meas_list
    boolean_vars_all = []
    feasible_point_mat_list = []

    if uda_meas_list:
        for true_indices in itertools.product(
            *[range(len(uda_meas.boolean_variables)) for uda_meas in uda_meas_list]
        ):

            for lv_meas, (true_idx, uda_meas) in enumerate(
                zip(true_indices, uda_meas_list)
            ):
                for lv_bool in range(len(uda_meas.boolean_variables)):
                    if lv_bool == true_idx:
                        uda_meas_list[lv_meas].boolean_variables[
                            lv_bool
                        ].true_value = True
                    else:
                        uda_meas_list[lv_meas].boolean_variables[
                            lv_bool
                        ].true_value = False

            for uda_meas in uda_meas_list:
                boolean_vars_all += uda_meas.boolean_variables

            for lv1 in range(n_random_x_points_per_bool_combo):
                for lv_var, var in enumerate(opt_variables):
                    var: LiftedQcQpVariable = var

                    X1 = rf.lift_feasible_point(
                        boolean_vars_all,
                        opt_variables,
                        column_names_all=lifted_variables_names_all,
                        formulation="with_x",
                    )
                    X1_arr = X1.toarray()
                    if len(X1_arr.shape) == 1:
                        X1_arr = X1_arr.reshape(1, -1)

                    feasible_point_mat_list.append(X1)
    else:
        for lv1 in range(n_random_x_points_per_bool_combo):
            for var in opt_variables:
                var: LiftedQcQpVariable = var

                var.true_value = var.generate_random()

                X1 = rf.lift_feasible_point(
                    boolean_vars_all,
                    opt_variables,
                    column_names_all=lifted_variables_names_all,
                    formulation="with_x",
                )
                X1_arr = X1.toarray()
                if len(X1_arr.shape) == 1:
                    X1_arr = X1_arr.reshape(1, -1)

                feasible_point_mat_list.append(X1)

    return feasible_point_mat_list


def var_type(var: str) -> str:
    if is_col_var(var):
        if "one" in var:
            return "one"
        else:
            return "bool"
    if not is_col_var(var):
        if "x" in var and "b" in var:
            return "bx"
        else:
            return "x"


def classify_redundant_constraints(var_tuples: List[List[PolyMatEntry]]):
    constraint_classes = []
    # Allowed variable classes
    # one, bool, bool_x

    for lv_constr in range(len(var_tuples)):
        print(f"Constraint {lv_constr}")
        for lv_var, var_tuple in enumerate(var_tuples[lv_constr]):
            var1 = var_tuple.var1
            var2 = var_tuple.var2
            val = var_tuple.val
            # print(f"{var1}, {var2} = {val}")
            # if set([var_type(var1), var_type[var2]]) == set(["one", "bool"]):
            # col1 =
            # if np.abs(val - 1) < 1e-3:
            # constraint_type = "diag"
