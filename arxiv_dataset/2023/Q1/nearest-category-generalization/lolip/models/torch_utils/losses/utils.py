import torch
from torch.distributions.dirichlet import Dirichlet
import numpy as np
from joblib import Parallel, delayed, parallel_backend

def get_trades_version(loss_name):
    if 't50trades' in loss_name:
        version = "top50"
    #elif 'attrades' in loss_name:
    #    version = "annealtop"
    elif 'sqtrades' in loss_name:
        version = "square"
    elif 'advhtrades' in loss_name:
        version = "advhard"
    elif 'htrades' in loss_name:
        version = "hard"
    elif 'mtrades' in loss_name:
        version = "max"
    elif 'wtrades' in loss_name:
        version = "weighted"
    else:
        version = None
    return version

def parse_num_steps(loss_name, base_name):
    if f"{base_name}10" in loss_name:
        ret = 10
    elif f"{base_name}20" in loss_name:
        ret = 20
    elif f"{base_name}40" in loss_name:
        ret = 40
    elif f"{base_name}100" in loss_name:
        ret = 100
    elif f"{base_name}1000" in loss_name:
        ret = 1000
    else:
        ret = 10
    return ret

def parse_loss_name(loss_name, base_name):
    if f"{base_name}10" in loss_name:
        ret = 10
    elif f"{base_name}20" in loss_name:
        ret = 20
    elif f"{base_name}100" in loss_name:
        ret = 100
    elif f"{base_name}200" in loss_name:
        ret = 200
    elif f"{base_name}1000" in loss_name:
        ret = 1000
    elif f"{base_name}16" in loss_name:
        ret = 16
    elif f"{base_name}32" in loss_name:
        ret = 32
    elif f"{base_name}6" in loss_name:
        ret = 6
    elif f"{base_name}4" in loss_name:
        ret = 4
    elif f"{base_name}3" in loss_name:
        ret = 3
    elif f"{base_name}2" in loss_name:
        ret = 2
    elif f"{base_name}.5" in loss_name:
        ret = .5
    elif f"{base_name}.1" in loss_name:
        ret = .1
    elif f"{base_name}1" in loss_name:
        ret = 1
    else:
        ret = 1
    return ret

def label_smoothing(y, gamma, n_classes, version="uniform", device="cuda"):
    batch_size = y.shape[0]
    y_onehot = torch.FloatTensor(batch_size, n_classes).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(batch_size, -1), 1)

    if version == "uniform":
        u = torch.ones_like(y_onehot).to(device) / n_classes
    elif version == "dirichlet":
        dirichlet = Dirichlet(torch.ones(n_classes))
        u = dirichlet.sample(torch.tensor([batch_size])).to(device)
    else:
        raise ValueError(f"[label smoothing] not supported version {version}.")

    ret = y_onehot * gamma + (1.-gamma) * u
    assert torch.allclose(ret.sum(1).cpu(), torch.ones(batch_size))
    return ret

def project_epsilon(epsilon, max_eps, x_natural=None):
    """
    max_eps: [batch_size, 1]
    """
    if isinstance(max_eps, float):
        epsilon = torch.clamp(epsilon, 0, max_eps)
    elif torch.is_tensor(max_eps):
        device = max_eps.get_device()
        epsilon = torch.min(epsilon, max_eps)
        epsilon = torch.max(epsilon, torch.zeros_like(epsilon).to(device))
    elif callable(epsilon[0]):
        def _helper(x, fn):
            return fn(x).reshape(-1)
        x_adv = x_natural + epsilon
        device = x_adv.get_device()
        x_adv = x_adv.cpu().numpy()
        with parallel_backend("loky", inner_max_num_threads=4):
            ret = Parallel(n_jobs=n_jobs)(delayed(_helper)(x, epsilon[i]) for i, x in enumerate(x_adv))
        ret = torch.from_numpy(np.array(ret)).float().to(device)
        return ret - x_natural
    else:
        raise ValueError
    return epsilon
