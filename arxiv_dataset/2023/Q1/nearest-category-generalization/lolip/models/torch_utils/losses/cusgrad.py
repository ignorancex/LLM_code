import torch
import torch.nn as nn
import torch.nn.functional as F

from ....attacks.torch.projected_gradient_descent import projected_gradient_descent
from .utils import parse_loss_name, parse_num_steps, label_smoothing, get_trades_version, project_epsilon


def get_cusgrad_stepsize(loss_name, max_eps):
    if 'v1au' in loss_name:
        eta = max_eps.max().item() / 20
    elif 'v2au' in loss_name:
        eta = max_eps.max().item() / 30
    elif 'v3au' in loss_name:
        eta = max_eps.max().item() / 50
    elif 'v1' in loss_name:
        eta = 0.01
    elif 'v2' in loss_name:
        eta = 0.05
    elif 'v3' in loss_name:
        eta = 0.1
    elif 'v4' in loss_name:
        eta = 0.02
    elif 'v5' in loss_name:
        eta = 0.5
    elif 'v6' in loss_name:
        eta = 1.0
    else:
        eta = 0.005
    return eta

def cusgrad_loss(loss_name, model, base_loss_fn, x, y, norm,
                 optimizer, current_eps, eps, n_classes, clip_min, clip_max):
    device = x.get_device()
    if 'trn' in loss_name:
        model.train()
    else:
        model.eval()
    steps = parse_num_steps(loss_name, 'K')

    eta = get_cusgrad_stepsize(loss_name, eps)
    c = 10

    current_eps = current_eps + eta
    current_eps = project_epsilon(current_eps, eps)

    advx = projected_gradient_descent(
        model, x, y=y, clip_min=clip_min, clip_max=clip_max,
        eps_iter=current_eps*2/steps, eps=current_eps, norm=norm, nb_iter=steps)

    kld = nn.KLDivLoss(reduction='batchmean')

    model.train()
    optimizer.zero_grad()
    outputs = model(advx)

    if 'linsmooth' in loss_name: # cusgradlinsmoothh
        dist = (advx - x).flatten(1).norm(p=norm, dim=1).unsqueeze(-1)
        smooth_factor = torch.clamp(1. - dist / eps, 0, 1)
        #smooth_factor = torch.min(
        #    (torch.ones(batch_size, 1)).float().to(device),
        #    (1. - dist / eps)
        #)
    else:
        smooth_factor = torch.clamp(1. - c * current_eps, 0, 1)

    if torch.is_tensor(smooth_factor):
        assert torch.all(smooth_factor <= 1), smooth_factor
        assert torch.all(smooth_factor >= 0), smooth_factor
    else:
        assert (smooth_factor <= 1) and (smooth_factor >= 0), smooth_factor

    yhat = label_smoothing(y, smooth_factor, n_classes, 'uniform', device)
    loss = kld(F.log_softmax(outputs, dim=1), yhat)

    if 'beta' in loss_name: # catbeta, catv1beta
        beta = parse_loss_name(loss_name, loss_name[loss_name.find('beta'):])
        loss = base_loss_fn(model(x), y) + beta * loss

    idx = (y != outputs.argmax(1)).clone().detach()
    current_eps = current_eps.clone().detach()
    current_eps.requires_grad_(False)
    if 'twotimes' in loss_name:
        if torch.is_tensor(eta):
            current_eps[idx] = (current_eps[idx] - 2*eta[idx])
        else:
            current_eps[idx] = (current_eps[idx] - 2*eta)
    else:
        if torch.is_tensor(eta):
            current_eps[idx] = (current_eps[idx] - eta[idx])
        else:
            current_eps[idx] = (current_eps[idx] - eta)
    current_eps = project_epsilon(current_eps, eps)

    return outputs, loss, current_eps
