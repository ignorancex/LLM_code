from typing import Tuple
from poly_matrix import PolyMatrix
from typing import List
from certifiable_uda_loc.symbolics import NamedMatrix
import numpy as np
from scipy import sparse as sp


def add_polymatrices(A1, A2):
    # Result stored in A1. A1 assumed to be symmetric
    if A1.symmetric and A2.symmetric:
        A1 = A1 + A2
    else:
        for var_i in A2.variable_dict_i.keys():
            for var_j in A2.variable_dict_j.keys():
                A1[var_i, var_j] = A1[var_i, var_j] + A2[var_i, var_j]
    return A1


def ij_th_element_of_MN_array(k: int, N: int) -> Tuple[int, int]:
    # Given an array with N columns,
    # return the 2D coordinates of the k'th element along it, column major format.
    j = k // N
    i = k - N * j
    return i, j


def kth_element_of_MN_array(i: int, j: int, N: int) -> int:
    # Given coordinates i, j of an array with N columns.
    # Find which index it is from, column major format.
    return N * j + i


def initialize_poly_matrix(
    var_names: List[str], sizes: List[Tuple] = None, vars_rows=None
) -> PolyMatrix:
    if sizes is None:
        sizes = [1] * len(var_names)
    if var_names == vars_rows:
        vars_rows = None

    if vars_rows is None:
        Q = PolyMatrix(symmetric=True)
        for i, var_name in enumerate(var_names):
            Q.add_variable_i(var_name, sizes[i])
        for i, var_name in enumerate(var_names):
            Q.add_variable_j(var_name, sizes[i])
    else:
        Q = PolyMatrix(symmetric=False)
        for i, var_name in enumerate(vars_rows):
            Q.add_variable_i(var_name, sizes[i])
        for i, var_name in enumerate(var_names):
            Q.add_variable_j(var_name, sizes[i])
    return Q


def initialize_matrix(vars_lifted, type: str):
    if type == "PolyMatrix":
        A = initialize_poly_matrix(vars_lifted)
        return A
    if type == "sympy":
        return NamedMatrix(vars_lifted, vars_lifted)


def coincident(one, two, thresh=1e-2):
    # https://stackoverflow.com/questions/41251911/check-if-an-array-is-a-multiple-of-another-array
    return (
        np.abs(
            np.dot(one, two) * np.dot(one, two) - np.dot(one, one) * np.dot(two, two)
        )
        < thresh
    )


def find_zero_columns(A: np.ndarray) -> List[int]:
    col_zero_idx_list = []
    for lv1 in range(A.shape[1]):
        if np.allclose(A[:, lv1], 0):
            col_zero_idx_list.append(lv1)
    return col_zero_idx_list


def find_linearly_dependent_columns(A: np.ndarray) -> List[int]:
    A = np.asarray(A)
    col_idx_LD = []
    for lv1 in range(A.shape[1]):
        col_1 = A[:, lv1]
        for lv2 in range(lv1 + 1, A.shape[1]):
            col_2 = A[:, lv2]
            if coincident(col_1, col_2):
                col_idx_LD.append(lv2)
    return col_idx_LD


def mat2vech(mat: np.array) -> np.array:
    sub_vector_list = []
    for lv1 in range(mat.shape[0]):
        current = mat[lv1, lv1:]  # pprint(moment_matrix1)
        sub_vector_list.append(current[0].reshape(-1, 1))
        sub_vector_list.append(np.sqrt(2) * current[1:].reshape(-1, 1))
    upper_half_vector = np.vstack(sub_vector_list).squeeze()
    return upper_half_vector


# TODO: dimension should be trivially inferred from the vector size..
def vech2mat(null_vec, N: int):
    """Convert vector represntation of matrix to matirx itself.
    Matrix is assumed symmetric so we only store upper half.
    vec size is size = n(n+1)/2

    [v1 v2 v3
    .   v4 v5
    .   .  v6]
    and so on.
    """
    # N_calced = int(-1 + np.sqrt(4 * 2 * len(null_vec)) / 2)
    # assert N == N_calced
    null_mat = np.zeros((N, N))
    vec_idx = 0
    for lv1 in range(N):
        for lv2 in range(lv1, N):  # upper half
            null_mat[lv1, lv2] = null_vec[vec_idx]
            if lv1 != lv2:
                null_mat[lv1, lv2] = null_vec[vec_idx] / np.sqrt(2)
            null_mat[lv2, lv1] = null_mat[lv1, lv2]
            vec_idx = vec_idx + 1
    return null_mat


def print_non_zero_entries(A: PolyMatrix, symmetric=False):
    lifted_variables_names_all = list(A.variable_dict_i.keys())

    str = ""
    for lv1 in range(len(lifted_variables_names_all)):
        if symmetric:
            start_point = lv1
        else:
            start_point = 0
        for lv2 in range(start_point, len(lifted_variables_names_all)):
            var1 = lifted_variables_names_all[lv1]
            var2 = lifted_variables_names_all[lv2]
            if np.abs(A[var1, var2]) > 1e-2:
                val = A[var1, var2]
                isint = np.abs(np.round(val) - val) < 1e-4
                if isint:
                    val = np.round(val)
                if (
                    var1 == "(th-T0-L0-meas-idx-0)_col1"
                    and var2 == "(th-T0-L0-meas-idx-1)_col1"
                ):
                    bop = 1
                str += f"{var1}, {var2} = {val.squeeze()}\n"
                print(f"{var1}, {var2} = {val.squeeze()}")
    return str


def np_to_polymatrix(
    vars_columns: List[str],
    vars_rows: List[str],
    np_matrix: np.ndarray,
    symmetric=True,
) -> PolyMatrix:
    if vars_columns == vars_rows and not symmetric:
        A_coo = sp.coo_array(np_matrix)
        A, _ = PolyMatrix.init_from_sparse(
            A_coo, var_dict={var: 1 for var in vars_columns}
        )
    else:
        A = initialize_poly_matrix(vars_columns, vars_rows=vars_rows)
        for i in range(np_matrix.shape[0]):
            for j in range(np_matrix.shape[1]):
                A[vars_rows[i], vars_columns[j]] = np_matrix[i, j]
    return A


def round_vec(v: np.ndarray, thresh=0.01):
    for lv1 in range(v.shape[0]):
        if np.abs(v[lv1]) < thresh:
            v[lv1] = 0
    return v
