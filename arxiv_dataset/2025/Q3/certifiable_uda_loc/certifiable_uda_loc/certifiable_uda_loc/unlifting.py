import pymlg
import certifiable_uda_loc.utils.string_utils as string_utils
import certifiable_uda_loc.constraints.constraints_boolean as constraints_boolean
import itertools
from typing import List
import numpy as np
from certifiable_uda_loc.variables import LiftedQcQpVariable
import certifiable_uda_loc.problem as rf
from py_factor_graph.data_associations import (
    VectorVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
)
import py_factor_graph.data_associations as pyfg_da
import py_factor_graph as pyfg
import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from typing import List, Tuple
from poly_matrix import PolyMatrix
from certifiable_uda_loc.utils.string_utils import var_to_colvar
import certifiable_uda_loc.utils.print_utils as print_utils
import certifiable_uda_loc.utils.matrix_utils as mat_utils
from typing import Union, Dict
import certifiable_uda_loc.variables as variables
import certifiable_uda_loc.lifting as lifting
from py_factor_graph.factor_graph import FactorGraphData


# TODO: Might have to edit this for 3d vs 2d later, SO2 hardcoded for now.
# TODO: Add pure vector variables.
def unlift_lifted_variables_to_factor_graph(
    fg: FactorGraphData,
    opt_variables: List[LiftedQcQpVariable],
    pose_to_lifted_var_indices: List[Tuple],
    landmark_to_lifted_var_indices: List[int],
) -> None:
    # This is saying that for pose lv1, the rotation and position variables respectively
    # are at indices contained in the tuple pose_to_lifted_var_indices[lv1]
    # of the lifted variable array.
    # Modifies fg by reference, no return type
    for lv1 in range(len(fg.pose_variables)):
        rot_lifted_var = opt_variables[pose_to_lifted_var_indices[lv1][0]]
        pos_lifted_var = opt_variables[pose_to_lifted_var_indices[lv1][1]]
        fg.pose_variables[lv1].estimated_position = (
            pos_lifted_var.estimated_value.reshape(
                -1,
            )
        )

        fg.pose_variables[lv1].estimated_theta = pymlg.SO2.Log(
            rot_lifted_var.estimated_value
        )
    for lv1 in range(len(fg.landmark_variables)):
        landmark_lifted_var = opt_variables[landmark_to_lifted_var_indices[lv1]]
        fg.landmark_variables[lv1].estimated_position = (
            landmark_lifted_var.estimated_value.reshape(
                -1,
            )
        )


def unlift(
    opt_variables: List[LiftedQcQpVariable],
    uda_measurements: List[UnknownDataAssociationMeasurement],
    X: PolyMatrix,
    hom_var: variables.HomogenizationVariable,
    thresh: float = 0.95,
) -> None:
    # Operation is done in place.
    # It modifies the vector variables and uda_measurements.
    nx = opt_variables[0].dims[0]
    hom_col_names = hom_var.column_names()
    for lv_opt_var in range(len(opt_variables)):
        col_names = opt_variables[lv_opt_var].column_names()
        val = np.zeros((nx, len(col_names)))
        for lv_col in range(len(col_names)):
            for lv_x in range(nx):
                val[lv_x, lv_col] = X[hom_col_names[lv_x], col_names[lv_col]]
        opt_variables[lv_opt_var].estimated_value = val

    for lv_meas in range(len(uda_measurements)):
        for lv_bool in range(len(uda_measurements[lv_meas].boolean_variables)):
            bool_var = uda_measurements[lv_meas].boolean_variables[lv_bool]
            bool_var_lifted = lifting.lift_scalar_variable(bool_var, nx)
            val = X[hom_col_names[0], bool_var_lifted.column_names()[0]]
            # val = X[hom_col_names[0], var_to_colvar(bool_var.name, 0)].squeeze()
            val = val > thresh
            val = val.squeeze()
            uda_measurements[lv_meas].boolean_variables[lv_bool].estimated_value = val
