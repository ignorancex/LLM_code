import torch
from torch.autograd import Variable

def _to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = _to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: _to_var(x), var)
        return var
