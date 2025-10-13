import navlie.lib.states as nv_states
from typing import List
import py_factor_graph.variables as pyfg_variables
import pymlg
import py_factor_graph.data_associations as pyfg_data_associations
from py_factor_graph.factor_graph import FactorGraphData
import py_factor_graph.measurements as pyfg_measurements
import numpy as np


def fg_poses_to_se2(pose_variables, true_or_est="true"):
    se2_states: List[nv_states.SE2State] = []
    for var in pose_variables:
        var: pyfg_variables.PoseVariable2D = var
        if true_or_est == "estimated":
            pose = pymlg.SE2.from_components(
                var.estimated_rotation_matrix,
                var.estimated_position,
            )
        if true_or_est == "true":
            pose = pymlg.SE2.from_components(
                var.true_rotation_matrix, var.true_position
            )

        se2_states.append(
            nv_states.SE2State(pose, state_id=var.name, stamp=var.timestamp)
        )
    return se2_states


def extract_unique_landmarks(fg: FactorGraphData):
    landmark_list = []
    for lv1, uda_meas in enumerate(fg.unknown_data_association_measurements):
        uda_meas: pyfg_data_associations.UnknownDataAssociationMeasurement = uda_meas
        for meas in uda_meas.measurement_list:
            meas: pyfg_measurements.PoseToKnownLandmarkMeasurement2D = meas
            add_meas = True
            for l in landmark_list:
                if np.linalg.norm(l - meas.r_a_la, 2) ** 2 < 1e-5:
                    add_meas = False
            if add_meas:
                landmark_list.append(meas.r_a_la)
    return landmark_list
