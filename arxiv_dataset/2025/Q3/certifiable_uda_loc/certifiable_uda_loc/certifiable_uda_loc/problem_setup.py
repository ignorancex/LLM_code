from typing import List
import certifiable_uda_loc.problem as rf
from typing import List, Tuple
from typing import Union, Dict
from py_factor_graph.factor_graph import FactorGraphData


def qcqp_optimization_variables(fg: FactorGraphData):

    opt_variables = []
    pose_to_lifted_var_indices: List[Tuple] = []
    landmark_to_lifted_var_indices: List[int] = []
    for var in fg.pose_variables:
        cur_vars = rf.lift_continuous_variable(var)
        rot_qcqp_var, pos_qcqp_var = cur_vars[0], cur_vars[1]
        opt_variables.append(rot_qcqp_var)
        opt_variables.append(pos_qcqp_var)
        # This is saying that for pose lv1, the rotation and position variables respectively
        # are at indices contained in the tuple pose_to_lifted_var_indices[lv1]
        # of the lifted variable array.
        pose_to_lifted_var_indices.append(
            (len(opt_variables) - 2, len(opt_variables) - 1)
        )

    for var in fg.landmark_variables:
        opt_variables += rf.lift_continuous_variable(var)
        landmark_to_lifted_var_indices.append(len(opt_variables) - 1)

    return opt_variables, pose_to_lifted_var_indices, landmark_to_lifted_var_indices
