import torch



hook_output = None

def layer_hook(module, input_, output):
    global hook_output
    hook_output = output


def access_activations_forward_hook(x, forward_function, forward_hook_point):
    handle = forward_hook_point.register_forward_hook(layer_hook)

    #with torch.no_grad():
    forward_function(*x)
    handle.remove()

    return hook_output.detach()#.cpu()



replace_hook_output = None

def layer_hook_replace(module, input_, output):
    return replace_hook_output


def get_pred_replace_activations(x, forward_function, forward_hook_point, replaced_activations):
    global replace_hook_output
    replace_hook_output = replaced_activations
    handle = forward_hook_point.register_forward_hook(layer_hook_replace)

    #with torch.no_grad():
    output = forward_function(*x)
    handle.remove()

    return output.detach()#.cpu()

