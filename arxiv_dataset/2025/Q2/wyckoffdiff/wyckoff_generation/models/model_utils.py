import torch


def split_number(number_to_split, max_value):
    q, r = divmod(number_to_split, max_value)
    result = [max_value] * q
    if r > 0:
        result.append(r)
    return result


def create_x_matrix(x_inf_dof, x_0_dof, zero_dof):
    x = torch.zeros(
        (x_inf_dof.shape[0] + x_0_dof.shape[0], x_inf_dof.shape[1] + 1),
        device=x_inf_dof.device,
    )
    x[zero_dof, 0] = x_0_dof.float()
    x[~zero_dof, 1:] = x_inf_dof.float()
    return x
