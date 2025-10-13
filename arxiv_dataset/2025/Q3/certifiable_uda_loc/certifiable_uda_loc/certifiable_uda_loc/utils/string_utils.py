from typing import Tuple


def var_to_colvar(var: str, col: int) -> str:
    return f"{var}_col{col}"


def colvar_to_var(colvar: str) -> Tuple[str, int]:
    var, col = colvar.split("_col")
    return var, int(col)


def get_var_column_names(var, dim, is_colvar):
    if is_colvar:
        return [var_to_colvar(var, lv1) for lv1 in range(dim)]
    else:
        return [var]


def bool_cont_var_name(bool_var: str, cont_var: str) -> str:
    if bool_var != "one":
        return f"{bool_var}_{cont_var}"
    else:
        return cont_var


def bool_cont_var_name_to_vars(bool_cont_var: str) -> Tuple[str, str]:
    return bool_cont_var.split("-")
