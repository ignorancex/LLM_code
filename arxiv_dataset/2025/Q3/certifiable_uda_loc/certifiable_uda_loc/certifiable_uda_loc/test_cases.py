import certifiable_uda_loc.problem as rf
from py_factor_graph.data_associations import (
    VectorVariable,
    PoseVariable2D,
    BooleanVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
)
import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from typing import List, Dict
from collections import namedtuple
from py_factor_graph.factor_graph import FactorGraphData
from certifiable_uda_loc.toy_example import (
    set_ground_truth_values,
)
import certifiable_uda_loc.problem as rf
from py_factor_graph.data_associations import (
    VectorVariable,
    PoseVariable2D,
    BooleanVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
)
import py_factor_graph.data_associations as pfg_mod_da
import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from typing import List, Dict
from collections import namedtuple
from py_factor_graph.factor_graph import FactorGraphData
from certifiable_uda_loc.toy_example import (
    set_ground_truth_values,
)
import py_factor_graph.variables as pfg_variables
import py_factor_graph.measurements as pfg_measurements
import pymlg
import py_factor_graph.measurements as pfg_measurements


def generate_random_toy_case(
    nx: int,
    n_components_per_factor: List[int],
    scale_center: float,
    scale_offset: float,
    scale_Q: float,
) -> FactorGraphData:
    vector_vars = VectorVariable(
        name="x", dims=(nx,), true_value=np.array([3] * nx).reshape(-1, 1)
    )

    uda_meas_list = []
    for i in range(len(n_components_per_factor)):
        measurement_list = []
        boolean_variables = []
        for j in range(n_components_per_factor[i]):
            center = np.random.uniform(0, scale_center, nx)
            offset = np.random.uniform(0, scale_offset)
            Q = np.random.uniform(0, scale_Q) * np.eye(nx)
            measurement_list.append(
                ToyExamplePrior(name="x", center=center, offset=offset, Q=Q)
            )
            bool_initial_val = False
            if j == 0:
                bool_initial_val = True

            boolean_variables.append(
                BooleanVariable(name=f"b{i}-{j}", true_value=bool_initial_val)
            )
        uda_meas_list.append(
            UnknownDataAssociationMeasurement(
                measurement_list=measurement_list, boolean_variables=boolean_variables
            )
        )

    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    set_ground_truth_values(fg)
    for lv1 in range(len(fg.vector_variables)):
        fg.vector_variables[lv1].estimated_value = -1000 * np.ones(
            fg.vector_variables[lv1].true_value.shape
        )
        pass
    return fg


def toy_unknown_data_association_cases() -> Dict[str, FactorGraphData]:
    case_dict = {}

    # dumbest base case
    nx = 1
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(name="x", center=np.array([2]), offset=1, Q=2)
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )

    case_dict["1d_1_meas"] = fg

    nx = 1
    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x",
                    center=np.array([21.95254016]),
                    offset=7.151893663724195,
                    Q=3.01381688,
                )
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["1d_1_meas_mc"] = fg

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(name="x", center=np.array([2]), offset=1, Q=2)
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["1d_1_meas"] = fg
    # Slightly less dumb case
    nx = 1
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(name="x", center=np.array([2]), offset=1, Q=2),
                ToyExamplePrior(name="x", center=np.array([3]), offset=0.5, Q=2),
            ],
            boolean_variables=[
                BooleanVariable(name="b0", true_value=False),
                BooleanVariable(name="b1", true_value=True),
            ],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["1d_1_meas_2_fac"] = fg

    # dimension up up up
    nx = 2
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3, 1]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x",
                    center=np.array([2, 3]),
                    offset=1,
                    Q=2 * np.eye(2),
                ),
                ToyExamplePrior(
                    name="x", center=np.array([4, 5]), offset=0.5, Q=2 * np.eye(2)
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="b0", true_value=False),
                BooleanVariable(name="b1", true_value=True),
            ],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["2d_1_meas_2_fac"] = fg

    nx = 1
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3, 4, 5]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x",
                    center=np.array([1]),
                    offset=1,
                    Q=2 * np.eye(nx),
                ),
                ToyExamplePrior(
                    name="x",
                    center=np.array([1]),
                    offset=2,
                    Q=2 * np.eye(nx),
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="ua0-b0", true_value=True),
                BooleanVariable(name="ua0-b1", true_value=True),
            ],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([10]), offset=5, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(
                    name="x", center=np.array([11]), offset=6, Q=2 * np.eye(nx)
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="ua1-b0", true_value=True),
                BooleanVariable(name="ua1-b1", true_value=True),
            ],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["1d_2_meas"] = fg

    nx = 2
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3, 4]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x",
                    center=np.array([1, 2]),
                    offset=1,
                    Q=2 * np.eye(nx),
                ),
                ToyExamplePrior(
                    name="x",
                    center=np.array([3, 4]),
                    offset=2,
                    Q=2 * np.eye(nx),
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="ua0-b0", true_value=True),
                BooleanVariable(name="ua0-b1", true_value=True),
            ],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([5, 6]), offset=5, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(
                    name="x", center=np.array([7, 8]), offset=6, Q=2 * np.eye(nx)
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="ua1-b0", true_value=True),
                BooleanVariable(name="ua1-b1", true_value=True),
            ],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["2d_2_meas"] = fg

    nx = 2
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3, 4]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x",
                    # center=np.array([16.68088019, 28.81297974]),
                    center=np.array([16, 28]),
                    offset=0,
                    Q=np.eye(2),
                ),
                ToyExamplePrior(
                    name="x",
                    center=np.array([1, 2]),
                    offset=2,
                    Q=np.eye(2),
                ),
            ],
            boolean_variables=[
                BooleanVariable(
                    name="b0-0",
                    true_value=True,
                    estimated_value=np.False_,
                    timestamp=None,
                ),
                BooleanVariable(
                    name="b0-1",
                    true_value=False,
                    estimated_value=np.False_,
                    timestamp=None,
                ),
            ],
            timestamp=None,
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x",
                    center=np.array([15, 20]),
                    offset=4,
                    Q=np.eye(2),
                ),
                ToyExamplePrior(
                    name="x",
                    center=np.array([8, 35]),
                    offset=0,
                    Q=np.eye(2),
                ),
            ],
            boolean_variables=[
                BooleanVariable(
                    name="b1-0",
                    true_value=True,
                    estimated_value=np.False_,
                    timestamp=None,
                ),
                BooleanVariable(
                    name="b1-1",
                    true_value=False,
                    estimated_value=np.False_,
                    timestamp=None,
                ),
            ],
            timestamp=None,
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["2d_2_meas_failing"] = fg

    nx = 1
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([0]), offset=1, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(name="x", center=np.array([1]), offset=2, Q=np.eye(nx)),
            ],
            boolean_variables=[
                BooleanVariable(name="ua0-b1", true_value=True),
                BooleanVariable(name="ua0-b2", true_value=False),
            ],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([0]), offset=3, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(name="x", center=np.array([1]), offset=2, Q=np.eye(nx)),
            ],
            boolean_variables=[
                BooleanVariable(name="ua1-b1", true_value=True),
                BooleanVariable(name="ua1-b2", true_value=False),
            ],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["3d_2_meas_simplified_more"] = fg

    nx = 1
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([0]), offset=1, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(name="x", center=np.array([1]), offset=2, Q=np.eye(nx)),
            ],
            boolean_variables=[
                BooleanVariable(name="ua0-b1", true_value=True),
                BooleanVariable(name="ua0-b2", true_value=False),
            ],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([0]), offset=3, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(name="x", center=np.array([1]), offset=2, Q=np.eye(nx)),
            ],
            boolean_variables=[
                BooleanVariable(name="ua1-b1", true_value=True),
                BooleanVariable(name="ua1-b2", true_value=False),
            ],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["3d_2_meas_simplified"] = fg

    nx = 3
    vector_vars = VectorVariable(name="x", dims=(nx,), true_value=np.array([3, 4, 5]))

    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([1, 2, 3]), offset=1, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(
                    name="x", center=np.array([4, 5, 6]), offset=2, Q=np.eye(nx)
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="ua0-b1", true_value=True),
                BooleanVariable(name="ua0-b2", true_value=False),
            ],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                ToyExamplePrior(
                    name="x", center=np.array([7, 8, 9]), offset=3, Q=2 * np.eye(nx)
                ),
                ToyExamplePrior(
                    name="x", center=np.array([10, 11, 12]), offset=2, Q=np.eye(nx)
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="ua1-b1", true_value=True),
                BooleanVariable(name="ua1-b2", true_value=False),
            ],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        vector_variables=[vector_vars],
    )
    case_dict["3d_2_meas"] = fg

    return case_dict


def generate_random_se2_uda_priors_case(
    n_components_per_factor: List[int],
    scale_pos: float = 10,
    mean_weight_rot: float = 0.5,
    mean_weight_pos: float = 0.2,
    scale_dev_weight_rot: float = 0.001,
    scale_dev_weight_pos: float = 0.001,
) -> FactorGraphData:
    nx = 2
    pose_var = pfg_variables.PoseVariable2D("T1", (1.1, 2.1), np.pi / 4)
    uda_meas_list = []

    for i in range(len(n_components_per_factor)):
        measurement_list = []
        boolean_variables = []
        for j in range(n_components_per_factor[i]):
            weight_rot = mean_weight_rot  # + scale_dev_weight_rot * np.random.random()
            weight_pos = mean_weight_pos  # + scale_dev_weight_pos * np.random.random()
            rot_angle = np.pi * np.random.random()
            # pos = scale_pos * np.random.random(2)
            pos = (scale_pos * np.random.random(), scale_pos * np.random.random())
            measurement_list.append(
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(rot_angle),
                    position=pos,
                    weight_rot=weight_rot,
                    weight_pos=weight_pos,
                )
            )
            bool_initial_val = False
            if j == 0:
                bool_initial_val = True

            boolean_variables.append(
                BooleanVariable(name=f"b{i}-{j}", true_value=bool_initial_val)
            )
        uda_meas_list.append(
            UnknownDataAssociationMeasurement(
                measurement_list=measurement_list, boolean_variables=boolean_variables
            )
        )

    fg = FactorGraphData(
        dimension=nx,
        unknown_data_association_measurements=uda_meas_list,
        pose_variables=[pose_var],
    )

    return fg


def pose_cases():
    case_dict = {}
    nx = 2
    pose_var = pfg_variables.PoseVariable2D("T1", (1.1, 2.1), np.pi / 4)
    prior_meas_list = [
        pfg_mod_da.PosePrior2D(
            name="T1",
            rotation=pymlg.SO2.Exp(0.5),
            position=(5, 6),
            weight_rot=0.5,
            weight_pos=0.1,
        )
    ]
    fg = FactorGraphData(
        dimension=nx, pose_variables=[pose_var], prior_pose_measurements=prior_meas_list
    )
    case_dict["single_pose_prior"] = fg

    nx = 2
    pose_var = pfg_variables.PoseVariable2D("T1", (1.1, 2.1), np.pi / 4)
    prior_meas_list = [
        pfg_mod_da.PosePrior2D(
            name="T1",
            rotation=pymlg.SO2.Exp(0.5),
            position=(5, 6),
            weight_rot=0.5,
            weight_pos=0.1,
        ),
        pfg_mod_da.PosePrior2D(
            name="T1",
            rotation=pymlg.SO2.Exp(1),
            position=(7, 8),
            weight_rot=0.9,
            weight_pos=0.2,
        ),
    ]
    fg = FactorGraphData(
        dimension=nx, pose_variables=[pose_var], prior_pose_measurements=prior_meas_list
    )
    case_dict["single_state_two_priors"] = fg

    nx = 2
    pose_var = pfg_variables.PoseVariable2D("T1", (1.1, 2.1), np.pi / 4)
    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(0.5),
                    position=(5, 6),
                    weight_rot=0.5,
                    weight_pos=0.1,
                )
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var],
        unknown_data_association_measurements=uda_meas_list,
    )
    case_dict["single_pose_prior_uda"] = fg

    nx = 2
    pose_var = pfg_variables.PoseVariable2D("T1", (1.1, 2.1), np.pi / 4)
    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(0.5),
                    position=(5, 6),
                    weight_rot=0.5,
                    weight_pos=0.1,
                )
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(1),
                    position=(10, 11),
                    weight_rot=0.5,
                    weight_pos=0.1,
                )
            ],
            boolean_variables=[BooleanVariable(name="b1", true_value=True)],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var],
        unknown_data_association_measurements=uda_meas_list,
    )
    case_dict["single_pose_two_prior_uda"] = fg

    nx = 2
    pose_var = pfg_variables.PoseVariable2D("T1", (1.1, 2.1), np.pi / 4)
    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(0.5),
                    position=(5, 6),
                    weight_rot=0.5,
                    weight_pos=0.1,
                ),
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(-0.5),
                    position=(1, 2),
                    weight_rot=0.2,
                    weight_pos=0.3,
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="b00", true_value=True),
                BooleanVariable(name="b01", true_value=False),
            ],
        ),
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(1),
                    position=(10, 11),
                    weight_rot=0.2,
                    weight_pos=0.1,
                ),
                pfg_mod_da.PosePrior2D(
                    name="T1",
                    rotation=pymlg.SO2.Exp(1),
                    position=(20, 30),
                    weight_rot=0.5,
                    weight_pos=0.2,
                ),
            ],
            boolean_variables=[
                BooleanVariable(name="b10", true_value=True),
                BooleanVariable(name="b11", true_value=False),
            ],
        ),
    ]
    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var],
        unknown_data_association_measurements=uda_meas_list,
    )
    case_dict["two_prior_two_factor_uda"] = fg

    return case_dict


def landmark_cases():
    case_dict = {}

    """
    %%%%%%%% TRIVIAL BASE CASE 1 POSE 1 LANDMARK 
    """
    nx = 2
    pose_name = "T1"
    landmark_name = "L1"
    pose_var = pfg_variables.PoseVariable2D(pose_name, (1.1, 2.1), np.pi / 4)
    landmark_var = pfg_variables.LandmarkVariable2D(landmark_name, (4.1, 5.1))
    prior_meas_list = [
        pfg_mod_da.PosePrior2D(
            name=pose_name,
            rotation=pymlg.SO2.Exp(0.0),
            position=(5, 6),
            weight_rot=0.5,
            weight_pos=0.1,
        ),
    ]
    pose2landmark_meas_list = [
        pfg_measurements.PoseToLandmarkMeasurement2D(
            pose_name=pose_name,
            landmark_name=landmark_name,
            x=3.1,
            y=5.1,
            translation_precision=0.1,
        )
    ]

    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var],
        landmark_variables=[landmark_var],
        pose_landmark_measurements=pose2landmark_meas_list,
        prior_pose_measurements=prior_meas_list,
    )

    case_dict["base_1p_1l"] = fg

    """
    %%%%%%%% TRIVIAL BASE CASE 1 POSE 1 LANDMARK - 1 MEASUREMENT 1 COMP - UDA  
    """

    nx = 2
    pose_name = "T1"
    landmark_name = "L1"
    pose_var = pfg_variables.PoseVariable2D(pose_name, (1.1, 2.1), np.pi / 4)
    landmark_var = pfg_variables.LandmarkVariable2D(landmark_name, (4, 5))
    # prior_meas_list = []
    prior_meas_list = [
        # pfg_mod_da.ToyExamplePrior(
        # landmark_name, np.array([5, 6]), offset=0.0, Q=0.3 * np.eye(2)
        # ),
        pfg_mod_da.PosePrior2D(
            name=pose_name,
            rotation=pymlg.SO2.Exp(0.0),
            position=(5, 6),
            weight_rot=0.5,
            weight_pos=0.1,
        ),
    ]
    pose2landmark_meas_list = []
    pose2landmark_meas_list = [
        pfg_measurements.PoseToLandmarkMeasurement2D(
            pose_name=pose_name,
            landmark_name=landmark_name,
            x=3.1,
            y=5.1,
            translation_precision=0.1,
        )
    ]
    # prior_meas_list = [
    #     pfg_mod_da.ToyExamplePrior(
    #         landmark_name, np.array([5, 6]), offset=0.0, Q=0.3 * np.eye(2)
    #     ),
    # ]
    uda_meas_list = []
    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_measurements.PoseToLandmarkMeasurement2D(
                    pose_name=pose_name,
                    landmark_name=landmark_name,
                    x=3.1,
                    y=5.1,
                    translation_precision=0.1,
                ),
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var],
        landmark_variables=[landmark_var],
        unknown_data_association_measurements=uda_meas_list,
        prior_pose_measurements=prior_meas_list,
        pose_landmark_measurements=pose2landmark_meas_list,
    )
    case_dict["trivial_base_case_uda"] = fg

    """
    %%%%%%%%%%%%%
    """

    nx = 2
    pose_name = "T1"
    landmark_name = "L1"
    pose_var = pfg_variables.PoseVariable2D(pose_name, (1.1, 2.1), np.pi / 4)
    landmark_var = pfg_variables.LandmarkVariable2D(landmark_name, (4, 5))
    prior_meas_list = [
        pfg_mod_da.ToyExamplePrior(
            landmark_name, np.array([8, 11]), offset=0.0, Q=0.3 * np.eye(2)
        ),
        pfg_mod_da.PosePrior2D(
            name=pose_name,
            rotation=pymlg.SO2.Exp(0.0),
            position=(5, 6),
            weight_rot=0.5,
            weight_pos=0.1,
        ),
    ]
    uda_meas_list = [
        UnknownDataAssociationMeasurement(
            measurement_list=[
                pfg_measurements.PoseToLandmarkMeasurement2D(
                    pose_name=pose_name,
                    landmark_name=landmark_name,
                    x=3,
                    y=5,
                    translation_precision=0.1,
                ),
                pfg_measurements.PoseToLandmarkMeasurement2D(
                    pose_name=pose_name,
                    landmark_name=landmark_name,
                    x=2,
                    y=10,
                    translation_precision=0.1,
                ),
            ],
            boolean_variables=[BooleanVariable(name="b0", true_value=True)]
            + [BooleanVariable(name="b1", true_value=False)],
        )
    ]
    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var],
        unknown_data_association_measurements=uda_meas_list,
        prior_pose_measurements=prior_meas_list,
        landmark_variables=[landmark_var],
    )
    case_dict["base_1p_1l_2meas"] = fg

    return case_dict


# Need to setup checks.
# Unlift, assertions, plots.
def relative_pose_cases():
    case_dict = {}

    """
    %%%%%%%% TRIVIAL BASE CASE 1 POSE 1 LANDMARK 
    """
    nx = 2
    pose1_name = "T1"
    pose2_name = "T2"
    pose_var1 = pfg_variables.PoseVariable2D(pose1_name, (0, 0), np.pi / 4)
    pose_var2 = pfg_variables.PoseVariable2D(pose2_name, (0, 0), np.pi / 4)
    prior_meas_list = [
        pfg_mod_da.PosePrior2D(
            name=pose1_name,
            rotation=pymlg.SO2.Exp(0.0),
            position=(5, 6),
            weight_rot=0.5,
            weight_pos=0.1,
        ),
    ]
    odom_measurement_list = [
        pfg_measurements.PoseMeasurement2D(
            pose1_name, pose2_name, 1.1, 2.1, 0.5, 0.1, 0.1
        )
    ]

    fg = FactorGraphData(
        dimension=nx,
        pose_variables=[pose_var1, pose_var2],
        prior_pose_measurements=prior_meas_list,
        odom_measurements=odom_measurement_list,
    )

    case_dict["base_2p"] = fg

    return case_dict
