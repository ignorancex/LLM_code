from typing import List
import itertools
import numpy as np


def create_discrete_feasible_point_vectors(
    lifted_variables: List[str],
    c_var_names_list_of_lists: List[List[str]],
):
    vec_list = []
    for true_indices in itertools.product(
        *[
            range(len(boolean_variables))
            for boolean_variables in c_var_names_list_of_lists
        ]
    ):
        vec = np.zeros(len(lifted_variables))
        vec[0] = 1  # Assumption that hom variable is in first spot.
        for lv_meas, (true_idx, boolean_variables) in enumerate(
            zip(true_indices, c_var_names_list_of_lists)
        ):
            for lv_bool in range(len(boolean_variables)):
                if lv_bool == true_idx:
                    for lv1, name in enumerate(lifted_variables):
                        if name == c_var_names_list_of_lists[lv_meas][lv_bool]:
                            vec[lv1] = 1
        vec_list.append(vec)
    return vec_list
