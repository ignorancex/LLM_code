from pymlg import SO2

from py_factor_graph.variables import (
    POSE_VARIABLE_TYPES,
    LANDMARK_VARIABLE_TYPES,
    PoseVariable2D,
    PoseVariable3D,
    LandmarkVariable2D,
)
from certifiable_uda_loc.monomials import generate_mixture_monomials
from typing import List, Union, Tuple, Dict
from poly_matrix import PolyMatrix
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
import numpy as np
import itertools

import certifiable_uda_loc.constraints.constraints_boolean as constraints_boolean
from certifiable_uda_loc.utils.string_utils import (
    var_to_colvar,
    get_var_column_names,
    colvar_to_var,
    bool_cont_var_name,
    bool_cont_var_name_to_vars,
)
from certifiable_uda_loc.utils.matrix_utils import np_to_polymatrix
from collections import namedtuple


def is_col_var(var_name: str) -> bool:
    return "_col" in var_name


def rotation_position_names_from_pose(pose_name: str):
    rot_name = f"{pose_name}-rot"
    pos_name = f"{pose_name}-pos"
    return rot_name, pos_name


class ColumnQcQcpVariable:
    def __init__(
        self,
        dims: int,
        name: str,
        parent_name: str = None,
        true_value: np.ndarray = None,
        estimated_value: np.ndarray = None,
    ):
        self.dims = dims
        self.name = name
        self.parent_name = parent_name
        self.true_value = true_value
        self.estimated_value = estimated_value

    def qcqpConstraintMatrices(self) -> List[PolyMatrix]:
        return []

    def column_names(self):
        return [self.name]


class LiftedQcQpVariable:
    def __init__(
        self,
        dims: Tuple[int],
        name: str,
        flat: bool = False,
        true_value=None,
        estimated_value=None,
    ):

        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims
        self.name = name
        self.flat = flat
        self.column_variables: List[ColumnQcQcpVariable] = []

        if len(dims) > 1 and dims[1] > 1:
            for lv1 in range(self.dims[1]):
                self.column_variables.append(
                    ColumnQcQcpVariable(
                        (dims[0], 1), var_to_colvar(name, lv1), parent_name=name
                    )
                )
        else:
            self.column_variables.append(
                ColumnQcQcpVariable((dims[0], 1), name, parent_name=name)
            )
        self.true_value = true_value
        self.estimated_value = estimated_value
        self.col_names = self.column_names()

    def generate_random(self):
        return np.random.random(self.dims)

    @property
    def true_value(self):
        return self.__true_value

    @true_value.setter
    def true_value(self, true_value):
        dims = self.dims
        self.__true_value = true_value
        if len(dims) > 1 and dims[1] > 1:
            for lv1 in range(self.dims[1]):
                true_value_lv1 = None
                if true_value is not None:
                    true_value_lv1 = true_value[:, lv1]
                self.column_variables[lv1].true_value = true_value_lv1
        else:
            if true_value is not None:
                self.column_variables[0].true_value = true_value.reshape(
                    -1,
                )

    @property
    def estimated_value(self):
        return self.__est_value

    @estimated_value.setter
    def estimated_value(self, est_value):
        dims = self.dims
        self.__est_value = est_value
        if len(dims) > 1 and dims[1] > 1:
            est_value_lv1 = None
            for lv1 in range(self.dims[1]):
                if est_value is not None:
                    est_value_lv1 = est_value[:, lv1]
                self.column_variables[lv1].estimated_value = est_value_lv1
        else:
            if est_value is not None:
                self.column_variables[0].estimated_value = est_value.reshape(
                    -1,
                )

    def column_names(self):
        return [var.name for var in self.column_variables]

    def qcqpConstraintMatrices(
        self,
        hom_var,
    ) -> List[PolyMatrix]:
        return []


class HomogenizationVariable(LiftedQcQpVariable):

    def qcqpConstraintMatrices(self) -> List[PolyMatrix]:
        var_names = [var.name for var in self.column_variables]

        if not self.flat:
            n = self.dims[1]
            A = initialize_poly_matrix(var_names)
            A[var_names[0], var_names[0]] = 1
            b = 1
            constraint_name = "hom_constraint"
            return [A], [b], [constraint_name]

    def unlift(self, var_value: np.ndarray):
        return var_value[0, 0]


class RotationVariable(LiftedQcQpVariable):

    def qcqpConstraintMatrices(
        self, hom_var: HomogenizationVariable
    ) -> List[PolyMatrix]:
        # Constraint Matrices that are associated with the variable itself
        # e.g. O(3) constraint for rotation matrices.
        col_names = self.column_names()
        hom_var_name = hom_var.column_names()[0]
        A_list = []
        for lv1 in range(len(col_names)):
            for lv2 in range(len(col_names)):
                A = initialize_poly_matrix(col_names)
                A[col_names[lv1], col_names[lv2]] = 1
                if lv1 == lv2:
                    A[hom_var_name, hom_var_name] = -1
                A_list.append(A)
        if self.dims[0] == 2:
            A = initialize_poly_matrix(col_names)
            A[col_names[0], hom_var.col_names[0]] = 1
            A[col_names[1], hom_var.col_names[1]] = -1
            A_list.append(A)
            A = initialize_poly_matrix(col_names)
            A[col_names[0], hom_var.col_names[1]] = 1
            A[col_names[1], hom_var.col_names[0]] = 1
            A_list.append(A)

        return A_list

    def generate_random(self):
        return SO2.random()
