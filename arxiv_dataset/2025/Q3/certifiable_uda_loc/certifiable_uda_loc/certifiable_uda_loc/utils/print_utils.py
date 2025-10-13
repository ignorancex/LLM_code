import pandas as pd
from typing import List
import numpy as np


def matprint(mat, fmt="g"):
    if isinstance(mat, np.matrix):
        mat = np.array(mat)
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def print_named_vector(var_names: List[str], vector: np.ndarray):
    pd.set_option("display.max_rows", None)
    df = pd.DataFrame(vector, index=var_names)
    print(df)


def custom_format(x):
    if x == 0:
        return "{:.1g}".format(x)  # One significant figure for zero
    else:
        return "{:.2f}".format(x)  # Two decimal places for other numbers


def pretty_print_array(X, rows=None, columns=None):
    if rows is None:
        rows = [f"r{lv1}" for lv1 in range(X.shape[0])]
    if columns is None:
        columns = [f"c{lv1}" for lv1 in range(X.shape[1])]
    df = pd.DataFrame(X, index=rows, columns=columns)
    formatted_df = df.applymap(custom_format)
    print(formatted_df)
