import pandas as pd
from certifiable_uda_loc.settings import (
    ProblemSolutionSettings,
    Se2CaseNoiseParameters,
)
import certifiable_uda_loc.utils.print_utils as print_utils

from certifiable_uda_loc.parsing import get_parser
import os
from pathlib import Path
from loguru import logger
from typing import List
import dill as pickle
from certifiable_uda_loc.trials import MonteCarloResultSdp, MonteCarloSummary
from matplotlib import pyplot as plt
import seaborn as sns
from collections import namedtuple
from certifiable_uda_loc.parsing import ParserArguments
from typing import Dict, Union
import itertools
from py_factor_graph.factor_graph import FactorGraphData
import pymlg
import navlie.utils.plot as nv_plot
import py_factor_graph.variables as pyfg_variables
import navlie.types as nv_types
import navlie.lib.states as nv_states
import py_factor_graph.measurements as pyfg_measurements
import navlie
from navlie.batch import gaussian_mixtures
import navlie.batch.residuals as nv_residuals
import navlie.lib.models as nv_models
import py_factor_graph.data_associations as pyfg_data_associations
import numpy as np
from certifiable_uda_loc.gauss_newton_baseline import (
    PriorResidualFro,
    RelativePoseResidualFro,
)


def convert_pyfg_to_navlie_factor_graph(fg: FactorGraphData):
    pose_keys = [var.name for var in fg.pose_variables]
    prior_residuals = []

    for prior_meas in fg.prior_pose_measurements:
        prior_meas: pyfg_data_associations.PosePrior2D = prior_meas
        T_check = pymlg.SE2.from_components(prior_meas.rotation, prior_meas.position)
        prior_state = nv_states.SE2State(value=T_check, stamp=0, state_id=pose_keys[0])
        prior_residuals.append(
            PriorResidualFro(
                pose_keys[0],
                prior_state,
                weight_rot=prior_meas.weight_rot,
                weight_pos=prior_meas.weight_pos,
                # weight_rot=1e6,
                # weight_pos=1e6,
            )
        )
    process_residuals = []
    for odom_meas in fg.odom_measurements:
        odom_meas: pyfg_measurements.PoseMeasurement2D = odom_meas
        # TODO: Figure out noises.
        dT = pymlg.SE2.from_components(
            odom_meas.rotation_matrix, odom_meas.translation_vector
        )
        resid = RelativePoseResidualFro(
            keys=[odom_meas.base_pose, odom_meas.to_pose],
            dT=dT,
            weight_rot=odom_meas.weight_rot,
            weight_pos=odom_meas.weight_pos,
        )
        process_residuals.append(resid)
    # TODO: For SLAM will have PoseToLandmark measurement (not known)
    # will need to add
    mixture_residuals = []
    for uda_meas in fg.unknown_data_association_measurements:
        uda_meas: pyfg_data_associations.UnknownDataAssociationMeasurement = uda_meas
        component_residuals = []
        for pyfg_meas in uda_meas.measurement_list:
            pyfg_meas: pyfg_measurements.PoseToKnownLandmarkMeasurement2D = pyfg_meas
            meas_model = nv_models.PointRelativePosition(
                landmark_position=pyfg_meas.r_a_la,
                R=np.eye(2) * 1 / pyfg_meas.weight,
            )
            nv_meas = nv_types.Measurement(pyfg_meas.r_b_lb, model=meas_model)
            resid = nv_residuals.MeasurementResidual([pyfg_meas.pose_name], nv_meas)
            component_residuals.append(resid)
        num_comp = len(component_residuals)

        mix_resid = gaussian_mixtures.MaxMixtureResidual(
            component_residuals, weights=[1 / num_comp] * num_comp
        )
        mixture_residuals.append(mix_resid)

    residuals = process_residuals + mixture_residuals + prior_residuals
    return residuals


def solve_max_mixture(
    fg: FactorGraphData,
    se2_states_init: List[nv_states.SE2State],
    include_max_mixture_residuals: bool = True,
    data_association_matrix_if_fixed: np.ndarray = None,
):
    se2_states = se2_states_init
    problem = navlie.batch.problem.Problem(solver="GN", step_tol=1e-8, max_iters=300)
    residuals = convert_pyfg_to_navlie_factor_graph(fg)
    for var in se2_states:
        var: nv_states.SE2State = var
        problem.add_variable(var.state_id, var)

    fixed_data_assocations = data_association_matrix_if_fixed is not None
    mm_resids = [
        res
        for res in residuals
        if isinstance(res, gaussian_mixtures.MaxMixtureResidual)
    ]

    if fixed_data_assocations:
        landmark_meas_resids = set_data_associations(
            mm_resids, data_association_matrix_if_fixed
        )
        for resid in landmark_meas_resids:
            problem.add_residual(resid)

    for resid in residuals:
        if not include_max_mixture_residuals or fixed_data_assocations:
            if isinstance(resid, gaussian_mixtures.MaxMixtureResidual):
                continue
            else:
                print(resid)
        problem.add_residual(resid)
    result_max_mix = problem.solve()

    # mm_resids: List[gaussian_mixtures.MaxMixtureResidual] = []
    # for resid in problem.residual_list:
    # if isinstance(resid, gaussian_mixtures.MaxMixtureResidual):
    # mm_resids.append(resid)

    return problem, result_max_mix


# How to use this to extract data associations after a GN convergence?
# Can I just use original max-mixture residuals??
def extract_data_associations_max_mixture(
    mm_resids: List[gaussian_mixtures.MaxMixtureResidual],
    se2_states_max_mixture: List[nv_states.SE2State],
):
    landmark_idx_per_meas_list = []
    for mm_resid in mm_resids:
        states_resid = []
        for key in mm_resid.keys:
            for state in se2_states_max_mixture:
                if state.state_id == key:
                    states_resid.append(state)
        (
            errors_by_component,
            _,
            _,
        ) = mm_resid.evaluate_component_residuals(states_resid, None)
        norms = np.array([np.linalg.norm(e, 2) ** 2 for e in errors_by_component])
        active_component = np.argmin(norms)
        landmark_idx_per_meas_list.append(active_component)
    # n_landmarks = len(np.unique(np.array(landmark_idx_per_meas_list)))
    if mm_resids:
        n_landmarks = len(mm_resids[0].errors)
    else:
        n_landmarks = 0
    data_association_matrix = np.zeros((len(mm_resids), n_landmarks))
    for meas_idx, landmark_idx in enumerate(landmark_idx_per_meas_list):
        data_association_matrix[meas_idx, landmark_idx] = 1

    # Need: Way to quantify data associations and associated error.
    # Need: Base example where stuff does not go well for max-mixture?

    # Can extract residuals from max-mixture....and find the minimum one.
    # Need to do the same for SDP solution & find which data association is active.
    # Need to label landmarks??
    return data_association_matrix


# TODO: The data association matrix way of doing it is not quite general..
# Is there a way to do this better
def set_data_associations(
    mm_resid_list: List[gaussian_mixtures.MaxMixtureResidual],
    data_association_matrix: np.ndarray,
):
    """
    Data association matrix: n_meas by n_landmarks
    """
    resid_list: List[nv_residuals.Residual] = []
    for lv_mm, mm_resid in enumerate(mm_resid_list):
        err_idx = np.argmin(np.abs(data_association_matrix[lv_mm, :] - 1))
        resid_list.append(mm_resid.errors[err_idx])
    return resid_list


#
