import sympy as smp
from typing import List
import numpy as np
import itertools
from typing import Tuple


def sp_dense_matrix(prefix, shape):

    if len(shape) == 1:
        return smp.Matrix([smp.symbols(f"{prefix}_{i}") for i in range(shape[0])])

    return smp.Matrix(
        [
            [smp.symbols(f"{prefix}_{i}{j}") for j in range(shape[1])]
            for i in range(shape[0])
        ]
    )


def symmetrize(mat):
    return (mat + mat.T) / 2


class NamedMatrix:
    def __init__(self, variables_i: List[str], variables_j: List[str], symmetric=True):
        self.nx = len(variables_i)
        self.ny = len(variables_j)
        self.slices_i = {key: idx for idx, key in enumerate(variables_i)}
        self.slices_j = {key: idx for idx, key in enumerate(variables_j)}
        self.symmetric = symmetric
        self.val = smp.zeros(self.nx, self.ny)
        self.variables_i = variables_i
        self.variables_j = variables_j

    def __setitem__(self, key_pair, value):
        if isinstance(key_pair, str):
            self.val[self.slices_i[key_pair]] = value
        else:
            self.val[self.slices_i[key_pair[0]], self.slices_j[key_pair[1]]] = value
            if self.symmetric:
                self.val[self.slices_j[key_pair[1]], self.slices_i[key_pair[0]]] = value

    def __getitem__(self, key_pair):
        if isinstance(key_pair, str):
            return self.val[self.slices_i[key_pair]]
        else:
            return self.val[self.slices_i[key_pair[0]], self.slices_j[key_pair[1]]]

    def set_submatrix_elements(
        self,
        vars_i: List[str],
        vars_j: List[str],
        submatrix: smp.Matrix,
    ):
        for lv1 in range(len(vars_i)):
            for lv2 in range(len(vars_j)):
                self.val[self.slices_i[vars_i[lv1]], self.slices_j[vars_j[lv2]]] = (
                    submatrix[lv1, lv2]
                )

    def symmetrize(self):
        self.val = symmetrize(self.val)


def massage_symbolic_matrix(A: smp.Matrix):
    A = smp.Matrix(A)
    for lv1 in range(A.shape[0]):
        for lv2 in range(A.shape[1]):
            if np.abs(A[lv1, lv2] - 0.5) < 1e-6:
                A[lv1, lv2] = smp.Rational(1, 2)
            if np.abs(A[lv1, lv2] + 0.5) < 1e-6:
                A[lv1, lv2] = -smp.Rational(1, 2)
            if np.abs(A[lv1, lv2] - 1) < 1e-6:
                A[lv1, lv2] = 1

    return A


def print_moment_matrix_constraint_as_expr(mat, moment_matrix):
    slice_list = []
    coeff_list = []
    expr = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            expr += mat[i, j] * moment_matrix[i, j]
            if mat[i, j] * moment_matrix[i, j] != 0:
                slice_list.append((moment_matrix[0, i], moment_matrix[0, j]))
                coeff_list.append(mat[i, j])
    print("Slices linked", slice_list)
    print("Coeffs", coeff_list)
    print(expr)
    print(smp.factor(expr))
    return expr


# TODO: Add polymatrix/numpy support to this.
def mat2vech(mat: smp.Matrix):
    sub_vector_list = []
    for lv1 in range(mat.shape[0]):
        current = mat[lv1, lv1:]  # pprint(moment_matrix1)
        sub_vector_list.append(current)

    upper_half_vector = smp.BlockMatrix(sub_vector_list).as_explicit()
    return upper_half_vector


# TODO: dimension should be trivially inferred from the vector size..
def vech2mat(null_vec, N: int):
    """Convert vector represntation of matrix to matirx itself.
    Matrix is assumed symmetric so we only store upper half.

    [v1 v2 v3
    .   v4 v5
    .   .  v6]
    and so on.
    """
    null_mat = smp.zeros(N, N)
    vec_idx = 0
    for lv1 in range(N):
        for lv2 in range(lv1, N):  # upper half
            null_mat[lv1, lv2] = null_vec[vec_idx]
            vec_idx = vec_idx + 1
    return null_mat


def generate_c_matrix(
    num_factor: int, num_components: int, active_component_indices: List[int]
):
    mat = np.zeros((num_factor, num_components))
    for lv1 in range(num_factor):
        mat[lv1, active_component_indices[lv1]] = 1
    return mat


def sub_vector_with_values(
    vec: smp.Matrix, c_symbol_lists: List[List[smp.Symbol]], c_value_mat: np.ndarray
) -> smp.Matrix:
    vec = vec.copy()
    num_factor = c_value_mat.shape[0]
    num_comp = c_value_mat.shape[1]
    for lv_fac in range(num_factor):
        for lv_comp in range(num_comp):
            vec = vec.subs(
                c_symbol_lists[lv_fac][lv_comp], int(c_value_mat[lv_fac, lv_comp])
            )

    return vec


def generate_feasible_point_symbolic(
    NUM_COMP: int, NUM_FACTOR: int, x_var_name: str
) -> Tuple[smp.Symbol, smp.Matrix, List[List[smp.Symbol]]]:
    if x_var_name is not None:
        x = smp.Symbol(x_var_name)
    else:
        x = None

    c_symbol_lists = [
        [
            smp.Symbol(f"c{lv_factor}{lv_comp}", positive=True)
            for lv_comp in range(NUM_COMP)
        ]
        for lv_factor in range(NUM_FACTOR)
    ]
    lifted_var_list = [1]
    for lv_factor in range(NUM_FACTOR):
        for lv_comp in range(NUM_COMP):
            lifted_var_list.append(c_symbol_lists[lv_factor][lv_comp])
    if x_var_name is not None:
        for lv_factor in range(NUM_FACTOR):
            for lv_comp in range(NUM_COMP):
                lifted_var_list.append(x * c_symbol_lists[lv_factor][lv_comp])
    x_lifted = smp.Matrix([lifted_var_list]).T
    return x, x_lifted, c_symbol_lists


def generate_feasible_vectorized_moment_matrix(NUM_COMP: int, NUM_FACTOR: int):

    x, x_lifted, c_symbol_lists = generate_feasible_point_symbolic(
        NUM_COMP, NUM_FACTOR, "x"
    )
    # From this moment we are considering vectorized versions of moment matrix.
    moment_matrix = x_lifted * x_lifted.T
    N = moment_matrix.shape[0]

    # pprint(moment_matrix)
    upper_half_vector = mat2vech(moment_matrix)

    c_subbed_vectors = []
    for active_component_indices in itertools.product(
        *([range(NUM_COMP)] * (NUM_FACTOR + 1))
    ):
        # print(active_component_indices)
        c_value_mat = generate_c_matrix(
            num_factor=NUM_FACTOR,
            num_components=NUM_COMP,
            active_component_indices=active_component_indices,
        )
        c_subbed_vectors.append(
            sub_vector_with_values(upper_half_vector, c_symbol_lists, c_value_mat).T
        )

    x_values = [1, 2, 3, 4]
    subbed_vectors = []
    for x_val in x_values:
        for c_subbed_vector in c_subbed_vectors:
            subbed_vector = c_subbed_vector.subs(x, x_val)
            subbed_vectors.append(subbed_vector)
    return subbed_vectors
