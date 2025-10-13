from certifiable_uda_loc.utils.matrix_utils import initialize_matrix
from typing import List
from poly_matrix import PolyMatrix
from certifiable_uda_loc.monomials import (
    monomial_variable_from_key_list,
    key_list_from_monomial_key,
)
import itertools
import collections
from typing import Dict


def constraint_one_variable_squared_equals_one(A_constraint: PolyMatrix) -> PolyMatrix:
    A_constraint["one", "one"] = 1
    return A_constraint


def equality_string_variables(lifted_var_1: str, lifted_var_2: str):
    arr1 = key_list_from_monomial_key(lifted_var_1, remove_ones=True)
    arr2 = key_list_from_monomial_key(lifted_var_2, remove_ones=True)
    variables_are_equal = collections.Counter(arr1) == collections.Counter(arr2)
    return variables_are_equal


def get_moment_constraint_dict(vars_lifted: str, hom_var_name: str = "one"):
    """Some entries in moment matrix are algebraically equal, must add constraint to enforce this equality
    in the relaxation"""
    print("Setting up moment constraints...")
    moment_matrix_strings = [
        [
            monomial_variable_from_key_list([vars_lifted[lv1], vars_lifted[lv2]])
            for lv2 in range(len(vars_lifted))
        ]
        for lv1 in range(len(vars_lifted))
    ]
    # flatten the list
    moment_matrix_strings = list(itertools.chain(*moment_matrix_strings))
    moment_matrix_variable_lists = [
        key_list_from_monomial_key(mom, hom_var_name) for mom in moment_matrix_strings
    ]
    moment_matrix_variable_lists = list(
        map(lambda x: tuple(sorted(x)), moment_matrix_variable_lists)
    )
    moment_matrix_variable_lists = list(dict.fromkeys(moment_matrix_variable_lists))
    moment_matrix_variable_lists = [list(mom) for mom in moment_matrix_variable_lists]

    # Create a dict based on these lists. Each key is a monomial product,
    # and the value is a list of pairs of monomials that give the monomial product in the key.
    moment_constraint_dict = {}
    for key_list in moment_matrix_variable_lists:
        if key_list == [hom_var_name]:
            continue
        moment_constraint_dict[monomial_variable_from_key_list(key_list)] = []

    for key in moment_constraint_dict.keys():
        # Loop through all the monomials.
        key_product_list = key_list_from_monomial_key(
            key, remove_ones=True, hom_var_name=hom_var_name
        )
        for lv1, mon1 in enumerate(vars_lifted):
            mon1_product_list = key_list_from_monomial_key(
                mon1, remove_ones=True, hom_var_name=hom_var_name
            )
            for lv2 in range(lv1, len(vars_lifted)):
                # for lv2 in range(len(monomial_key_list)):
                mon2 = vars_lifted[lv2]
                if mon1 == mon2:
                    continue
                product_list = mon1_product_list + key_list_from_monomial_key(
                    mon2, remove_ones=True, hom_var_name=hom_var_name
                )

                # Collections counter way of doing it.. proper way but takes time
                # equal_product = collections.Counter(
                # product_list
                # ) == collections.Counter(key_product_list)
                # Set way that assumes the elements in lists are uniqe.
                equal_product = set(product_list) == set(key_product_list)
                # Now compare.
                if equal_product:
                    moment_constraint_dict[key].append((mon1, mon2))

    keys_to_delete = []
    # The case with the "one" is particular. Treat it separately..
    moment_constraint_dict_ones: Dict[List] = {}
    for key in moment_constraint_dict.keys():
        if "_one" not in key:
            continue
        if key[-4:] == "_one":
            key_check = key[:-4]
        if "_one_" in key:
            key_check = key.replace("_one_", "_")

        if key_check in moment_constraint_dict_ones:
            moment_constraint_dict_ones[key_check] += moment_constraint_dict[key]
        else:
            moment_constraint_dict_ones[key_check] = moment_constraint_dict[key]

    for key1 in moment_constraint_dict:
        for key2 in moment_constraint_dict_ones:
            if key1 == key2:
                moment_constraint_dict[key1] += moment_constraint_dict_ones[key2]

    # Remove potential duplicates in moment constraint lists
    for key in moment_constraint_dict.keys():
        deletion_indices = []
        for lv0, pair0 in moment_constraint_dict[key]:
            for lv1, pair1 in moment_constraint_dict[key]:
                if lv0 == lv1:
                    continue
                else:
                    if pair0 == pair1:
                        if lv1 > lv0:
                            deletion_indices.append(lv1)
        moment_constraint_dict[key] = [
            mom
            for mom, del_mom in zip(moment_constraint_dict[key], deletion_indices)
            if not del_mom
        ]

    for key in moment_constraint_dict.keys():
        if len(moment_constraint_dict[key]) < 2:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del moment_constraint_dict[key]

    for key in moment_constraint_dict.keys():
        print(key)
    print("Done setting up moment constraints dict")
    return moment_constraint_dict


def get_moment_constraints(
    vars_lifted: List[str], type: str = "PolyMatrix", hom_var_name: str = "one"
) -> List[PolyMatrix]:
    # This is bugged somewhere. Messes up the problem solution.
    moment_constraint_dict = get_moment_constraint_dict(vars_lifted, hom_var_name)
    A_list: List[PolyMatrix] = []
    for _, equal_variable_list in moment_constraint_dict.items():
        for pair in list(itertools.combinations(equal_variable_list, 2)):
            # print("Pair 0", pair[0], "Pair 1", pair[1])
            A = initialize_matrix(vars_lifted, type)
            A[pair[0][0], pair[0][1]] = 1
            A[pair[1][0], pair[1][1]] = -1
            A_list.append(A)
    return A_list
