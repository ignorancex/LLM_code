import numpy as np
from typing import List


class NamedMatrixNumpy:
    """A class to work with 2Dnumpy matrices that have named rows and columns."""

    def __init__(
        self,
        variables_i: List[str],
        variables_j: List[str] = ["_"],
        val: np.array = None,
    ):
        self.variables_i = variables_i
        self.variables_j = variables_j

        self.nx = len(variables_i)
        self.ny = len(variables_j)
        self.slices_i = {key: idx for idx, key in enumerate(variables_i)}
        self.slices_j = {key: idx for idx, key in enumerate(variables_j)}
        if val is None:
            self.val = np.zeros((self.nx, self.ny))
        else:
            self.val = np.real_if_close(val)

    def transpose(self):
        return NamedMatrixNumpy(
            variables_i=self.variables_j, variables_j=self.variables_i, val=self.val.T
        )

    def __setitem__(self, key_pair, value):
        if isinstance(key_pair, str):
            self.val[self.slices_i[key_pair]] = value
        else:
            self.val[self.slices_i[key_pair[0]], self.slices_j[key_pair[1]]] = value

    def __getitem__(self, key_pair):
        if isinstance(key_pair, str):
            return self.val[self.slices_i[key_pair]]
        else:
            return self.val[self.slices_i[key_pair[0]], self.slices_j[key_pair[1]]]

    def set_submatrix_elements(
        self,
        vars_i: List[str],
        vars_j: List[str],
        submatrix: np.array,
    ):
        for lv1 in range(len(vars_i)):
            for lv2 in range(len(vars_j)):
                self.val[self.slices_i[vars_i[lv1]], self.slices_j[vars_j[lv2]]] = (
                    submatrix[lv1, lv2]
                )

    def pretty_print(self, fmt="g"):
        # Modified from https://gist.github.com/lbn/836313e283f5d47d2e4e
        mat = self.val
        if self.ny == 1:
            mat = mat.reshape(-1, 1)
        col_var_lengths = [len(var) for var in self.variables_j]

        col_maxes = [
            max(
                [len(("{:" + fmt + "}").format(x)) for x in col]
                + [col_var_lengths[lv_col]]
            )
            for lv_col, col in enumerate(mat.T)
        ]

        row_var_max = max([len(x) for x in self.variables_i])
        print("".ljust(row_var_max), end="  ")
        for lv_col, var_j in enumerate(self.variables_j):
            print(var_j.ljust(col_maxes[lv_col]), end="  ")
        print("")

        for lv_row, x in enumerate(mat):
            print(self.variables_i[lv_row].ljust(row_var_max), end="  ")
            for i, y in enumerate(x):
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
            print("")


if __name__ == "__main__":
    # Example usage
    vars_i = ["ax", "bx", "cx"]
    vars_j = ["ay", "by", "cy"]
    val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    named_mat = NamedMatrixNumpy(vars_i, vars_j, val)
    named_mat.pretty_print()

    # Set a specifc value
    print("Setting named_mat[ax, by] = 10")
    named_mat[["ax", "by"]] = 10
    named_mat.pretty_print()

    # Set a subset of named_mat to a specific value
    print("Modify a subblock")
    named_mat.set_submatrix_elements(
        ["ax", "bx"], ["ay", "by"], np.array([[11, 12], [13, 14]])
    )
    named_mat.pretty_print()

    # Vectors are supported as column matrices,
    # with an added dummy column
    print("Vector as column matrix")
    vars_i = ["ax", "bx", "cx"]
    vars_j = ["_"]
    val = np.array([1, 2, 3])
    named_mat = NamedMatrixNumpy(vars_i, vars_j, val)
    named_mat.pretty_print()
