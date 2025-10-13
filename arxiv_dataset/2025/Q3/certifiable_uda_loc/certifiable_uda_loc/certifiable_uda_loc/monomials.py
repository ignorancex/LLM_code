from typing import List


def monomial_variable_from_key_list(key_list: List[str]):
    return "_".join(key_list)


def key_list_from_monomial_key(
    key: str, remove_ones: bool = False, hom_var_name: str = "one"
):
    key_list = key.split("_")
    if remove_ones:
        key_list = [x for x in key_list if key_list != hom_var_name]
    return key_list


def generate_mixture_monomials(
    c_var_names: List[str], x_var_names: List[str], type_list=["c", "cx"]
):
    # type_list=["x", "c", "cx"]

    monomial_key_list = ["one"]
    monomial_key_list_of_lists = [["one"]]

    if "c" in type_list:
        monomial_key_list += c_var_names
        monomial_key_list_of_lists += [[var] for var in c_var_names]
    if "x" in type_list:
        monomial_key_list += x_var_names
        monomial_key_list_of_lists += [[var] for var in x_var_names]
    for c_var in c_var_names:
        for x_var in x_var_names:
            if "cx" in type_list:
                monomial_key_list_of_lists.append([c_var, x_var])
                monomial_key_list.append(
                    monomial_variable_from_key_list([c_var, x_var])
                )
    return monomial_key_list, monomial_key_list_of_lists
