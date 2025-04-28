# -*- coding: utf-8 -*-
"""Implement the corrections table basis rotation"""
import json

import numpy as np
import pandas as pd

from .compute_theory import flatten


def rotate_to_fit_basis(lin_dict, quad_dict, rotation_matrix_path):
    """
    Rotate to fitting basis

    Parameters
    ----------
        lin_dict: dict
            theory dictionary with linear operator corrections in the table
            basis
        quad_dict: dict
            theory dictionary with quadratic operator corrections in the table
            basis, emptry if quadratic corrections are not used
        rotation_matrix: pandas.DataFrame
            rotation matrix from tables basis to fitting basis

    Returns
    -------
        lin_dict_fit_basis: dict
            theory dictionary with linear operator corrections in the fit
            basis
        quad_dict_fit_basis: dict
            theory dictionary with quadratic operator corrections in the fit
            basis, emptry if quadratic corrections are not used
    """
    with open(rotation_matrix_path, encoding="utf-8") as f:
        rot = json.load(f)
    rotation_matrix = pd.DataFrame(
        data=rot["matrix"], index=rot["ypars"], columns=rot["xpars"]
    )

    # select corrections to keep
    def is_to_keep(op1, op2=None):
        if op2 is None:
            return op1 in rotation_matrix.columns
        return op1 in rotation_matrix.columns and op2 in rotation_matrix.columns

    lin_dict_to_keep = {k: val for k, val in lin_dict.items() if is_to_keep(k)}
    lin_df = pd.DataFrame(lin_dict_to_keep)

    # select the columns corresponding to the
    # relevant corrections for the operator card basis
    R = rotation_matrix[lin_df.columns]
    lin_dict_fit_basis = lin_df @ R.T

    # look at the quadratic corrections?
    quad_dict_fit_basis = {}
    if quad_dict == {}:
        return lin_dict_fit_basis.to_dict("list"), quad_dict_fit_basis

    quad_dict_to_keep = {
        k: val
        for k, val in quad_dict.items()
        if is_to_keep(k.split("*")[0], k.split("*")[1])
    }
    tensor = []
    # loop over table basis pairs
    # and build an (n_op_table, n_dat, n_op_fit, n_op_fit) tensor

    for col, values in quad_dict_to_keep.items():
        o1, o2 = col.split("*")
        r1 = rotation_matrix[o1]
        r2 = rotation_matrix[o2]
        r1r2 = np.outer(r1, r2)
        r1r2o1o2 = np.einsum("i,kj->ikj", values, r1r2)
        tensor.append(r1r2o1o2)
    # sum over table basis entries
    tensor = np.array(tensor)
    new_quad_corrections = tensor.sum(axis=0)

    # flatten the tensor (n_dat, n_op_fit, n_op_fit) -> (n_dat, n_op_fit_pairs)
    new_quad_matrix = []
    for correction in new_quad_corrections:
        new_quad_matrix.append(flatten(correction, axis=1))
    new_quad_matrix = np.array(new_quad_matrix)

    matrix_new_keys = []
    for r1 in rotation_matrix.index:
        row = []
        for r2 in rotation_matrix.index:
            row.append(f"{r1}*{r2}")
        matrix_new_keys.append(row)

    new_keys = flatten(np.array(matrix_new_keys))

    for i, new_key in enumerate(new_keys):
        quad_dict_fit_basis[new_key] = new_quad_matrix[:, i]

    return lin_dict_fit_basis.to_dict("list"), quad_dict_fit_basis
