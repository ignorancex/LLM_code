import torch
from copy import deepcopy


def state_to_tensor(model_state_dict):
    """
    Convert a state dict to a concatenated tensor
    Note: it is deep copy, since torch.cat is deep copy
    :param model_state_dict:
    :return:
    """
    tensor = [t.view(-1) for t in model_state_dict.values()]
    tensor = torch.cat(tensor)
    return tensor


def tensor_to_state(tensor, model_state_dict_template):
    """
    Convert a tensor back to state dict.
    Note: apply deepcopy inside the function. Only use the input state dict as a template
    :param model_state_dict:
    :return:
    """
    curr_idx = 0
    model_state_dict = deepcopy(model_state_dict_template)
    for key, value in model_state_dict.items():
        numel = value.numel()
        shape = value.shape
        model_state_dict[key].copy_(tensor[curr_idx:curr_idx + numel].view(shape))
        curr_idx += numel

    return model_state_dict


def model_numel(model, typ='all'):
    """
    Calculate the number of parameters of a model
    :param model:
    :return:
    """
    num = 0
    if typ == 'all':
        for tensor in model.state_dict().values():
            num += tensor.numel()
    elif typ == 'uploaded':
        for tensor in model.uploaded_state_dict().values():
            num += tensor.numel()
    else:
        raise NotImplementedError('Unknown type of parameter to count. ')

    return num


