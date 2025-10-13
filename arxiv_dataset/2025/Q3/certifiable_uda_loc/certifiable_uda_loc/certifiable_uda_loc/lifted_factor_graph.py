import certifiable_uda_loc.lifting as lifting
from py_factor_graph.factor_graph import FactorGraphData
import certifiable_uda_loc.problem as rf
import certifiable_uda_loc.variables as variables
import numpy as np
from typing import List, Tuple, Dict
from poly_matrix import PolyMatrix
import itertools
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
import py_factor_graph.data_associations as pyfg_da


# Can we include this in lifted factor graph?
class UdaMeasurementGroup:
    def __init__(
        self,
        uda_meas_list: List[pyfg_da.UnknownDataAssociationMeasurement],
        opt_variables: List[variables.LiftedQcQpVariable],
    ):
        self.uda_meas_list: List[pyfg_da.UnknownDataAssociationMeasurement] = (
            uda_meas_list
        )
        self.opt_variables: List[str] = opt_variables
        self.cont_column_variables: List[variables.LiftedQcQpVariable] = None
        self.boolean_variables: List[pyfg_da.BooleanVariable] = None
        self.c_var_list_of_lists: List[List[str]] = None
        self.c_constraints_per_factor: List[PolyMatrix] = []
        self.c_constraints_interfactor: List[PolyMatrix] = []
        self.c_constraints_all: List[PolyMatrix] = []
        self.initialize_internal_variables()

    def reorder_column_variables_based_on_top_level(self, lifted_variables_all):
        # Theres a bit of randomness introduced in the ordering of self.cont_column_variables
        # due to use of set in the initialziation. So we remove this randomness for cleanliness.
        new_col_vars = []
        for var in lifted_variables_all:
            if var in self.cont_column_variables:
                new_col_vars.append(var)
        self.cont_column_variables = new_col_vars

    def initialize_internal_variables(
        self,
    ):
        cont_column_variables = []
        c_var_list_of_lists = []
        for uda_meas in self.uda_meas_list:
            cont_column_variables += lifting.get_uda_meas_variables(uda_meas)
            c_var_list_of_lists.append([var.name for var in uda_meas.boolean_variables])
        self.cont_column_variables = list(set(cont_column_variables))

        self.c_var_list_of_lists = c_var_list_of_lists
        self.boolean_variables = []
        for uda_meas in self.uda_meas_list:
            self.boolean_variables += uda_meas.boolean_variables


class LiftedFactorGraph:
    def __init__(
        self,
        fg: FactorGraphData,
        sparse_bool_cont_variables: bool = False,
        use_locking_constraint=True,
    ):

        self.fg = fg
        self.sparse_bool_cont_variables = sparse_bool_cont_variables
        self.use_locking_constraint = use_locking_constraint
        self.locked_pose_idx = None
        self.locked_pose_variables: List[variables.LiftedQcQpVariable] = []
        self.opt_variables: List[variables.LiftedQcQpVariable] = []
        self.constraint_dict_all: Dict[str, List[PolyMatrix]] = {}
        self.nx: int = None

        self.pose_to_lifted_var_indices: List[Tuple] = []
        self.landmark_to_lifted_var_indices: List[int] = []
        self.vector_to_lifted_var_indices: List[int] = []

        self.cont_column_variables = (
            []
        )  # these are the continuous column variables.. # TODO rename to something less confusing

        self.hom_var: variables.HomogenizationVariable = None

        self.lifted_rot_variables: List[variables.RotationVariable] = []
        self.lifted_pos_variables: List[variables.LiftedQcQpVariable] = []
        self.lifted_vector_variables: List[variables.LiftedQcQpVariable] = []
        self.bool_lifted_vars: List[variables.LiftedQcQpVariable] = []
        self.bool_cont_lifted_vars: List[variables.LiftedQcQpVariable] = []

        self.standard_meas_list = (
            fg.prior_pose_measurements
            + fg.pose_landmark_measurements
            + fg.odom_measurements
            + fg.toy_example_priors
            + fg.landmark_priors
        )
        self.uda_meas_list = self.fg.unknown_data_association_measurements
        self.uda_meas_dict: Dict[
            Tuple[str], List[pyfg_da.UnknownDataAssociationMeasurement]
        ] = None
        self.uda_group_list_sparse: List[UdaMeasurementGroup]
        self.uda_group_list_dense: List[UdaMeasurementGroup]
        self.initialize_lifted_continuous_variables()
        self.initialize_column_variables()
        self.initialize_all_variable_names_from_uda()
        self.initialize_discrete_variables()
        self.initialize_uda_meas_dict()
        self.initialize_uda_groups()

    def initialize_uda_meas_dict(
        self,
    ) -> None:
        # Breaks down the uda meas list into chunks, each of which corresponds to the same state subset.
        uda_meas_dict = {}
        for uda_meas in self.uda_meas_list:
            column_variables = lifting.get_uda_meas_variables(uda_meas)
            key = tuple(column_variables)
            if key in uda_meas_dict:
                uda_meas_dict[key].append(uda_meas)
            else:
                uda_meas_dict[key] = [uda_meas]
        self.uda_meas_dict = uda_meas_dict

    def initialize_uda_groups(self):
        uda_group_list_sparse: List[UdaMeasurementGroup] = []
        for key, uda_meas_list in self.uda_meas_dict.items():
            opt_variables_lv1 = []
            for var in self.opt_variables:
                col_names = var.col_names
                if any([name in key for name in col_names]):
                    opt_variables_lv1.append(var)
            uda_group = UdaMeasurementGroup(uda_meas_list, opt_variables_lv1)
            uda_group.reorder_column_variables_based_on_top_level(
                self.lifted_variables_names_all
            )

            # uda_group.cont_variables_top_level = key

            uda_group_list_sparse.append(uda_group)

        uda_group_dense = UdaMeasurementGroup(self.uda_meas_list, self.opt_variables)
        self.uda_group_list_sparse = uda_group_list_sparse
        self.uda_group_list_dense = [uda_group_dense]

    def initialize_lifted_continuous_variables(self):
        self.locked_pose_idx = 0
        for lv1, var in enumerate(self.fg.pose_variables):
            cur_vars = rf.lift_continuous_variable(var)
            rot_qcqp_var, pos_qcqp_var = cur_vars[0], cur_vars[1]
            self.opt_variables.append(rot_qcqp_var)
            self.opt_variables.append(pos_qcqp_var)
            if lv1 == self.locked_pose_idx:
                self.locked_pose_variables += [rot_qcqp_var, pos_qcqp_var]
            self.pose_to_lifted_var_indices.append(
                (len(self.opt_variables) - 2, len(self.opt_variables) - 1)
            )

        for var in self.fg.landmark_variables:
            self.opt_variables += rf.lift_continuous_variable(var)
            self.landmark_to_lifted_var_indices.append(len(self.opt_variables) - 1)

        for var in self.fg.vector_variables:
            self.opt_variables += rf.lift_continuous_variable(var)
            self.vector_to_lifted_var_indices.append(len(self.opt_variables) - 1)

        self.nx = self.opt_variables[0].dims[0]
        self.hom_var = variables.HomogenizationVariable(
            (self.nx, self.nx), "one", true_value=np.eye(self.nx)
        )

    def initialize_column_variables(self):
        self.cont_column_variables: List[variables.ColumnQcQcpVariable] = []
        for var in self.opt_variables:
            var: variables.LiftedQcQpVariable
            for lv1, column_var in enumerate(var.column_variables):
                column_var: variables.ColumnQcQcpVariable = column_var
                self.cont_column_variables.append(column_var)
                assert np.allclose(column_var.true_value, var.true_value[:, lv1])

    def initialize_all_variable_names_from_uda(self):
        _, self.lifted_variables_names_all = lifting.get_lifted_variables_all(
            self.hom_var,
            self.cont_column_variables,
            self.uda_meas_list,
            sparse_bool_cont_variables=self.sparse_bool_cont_variables,
        )

    def get_continuous_variable_constraints(self):
        A_dict_xi_by_variable = {}

        locked_var_names = [var.name for var in self.locked_pose_variables]
        for var in self.opt_variables:
            if var.name in locked_var_names and self.use_locking_constraint:
                continue
            A_list = var.qcqpConstraintMatrices(self.hom_var)
            key = tuple(var.col_names)
            A_dict_xi_by_variable[key] = A_list

        rot_vars = self.lifted_rot_variables
        lifted_variables_names_all = self.lifted_variables_names_all

        A_dict_xi_intervariable = []
        # Might need generalization to SO(3).. this is just the orthonormality constraint of the resulting DCM innit?
        for var1, var2 in itertools.combinations(rot_vars, 2):
            key = tuple(list(var1.col_names + var2.col_names))
            A_dict_xi_intervariable[key] = []
            A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
            A[var1.col_names[0], var2.col_names[1]] = 1
            A[var2.col_names[0], var1.col_names[1]] = 1
            A_dict_xi_intervariable[key].append(A)

            A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
            A[var1.col_names[0], var2.col_names[0]] = 1
            A[var2.col_names[1], var1.col_names[1]] = -1
            A_dict_xi_intervariable[key].append(A)

        return A_dict_xi_by_variable, A_dict_xi_intervariable

    def initialize_discrete_variables(self):

        boolean_vars_all = []
        for uda_meas in self.uda_meas_list:
            boolean_vars_all += uda_meas.boolean_variables

        self.bool_lifted_vars, self.bool_cont_lifted_vars = (
            lifting.lift_data_association_variables(
                boolean_vars_all, self.opt_variables
            )
        )

    def __repr__(self):
        str = f"Lifted Factor Graph with Properties\n"
        str += f"Number of lifted vector variables {len(self.lifted_variables_names_all)} \n"
        return str
