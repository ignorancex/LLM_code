
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trades import trades_loss, project_delta
from .cusgrad_trades import cusgradtrades_loss
from ....attacks.torch.projected_gradient_descent import projected_gradient_descent
from .utils import parse_loss_name, parse_num_steps, label_smoothing, get_trades_version, project_epsilon
from .cusgrad import get_cusgrad_stepsize
from .cusgrad_trades_ellipsoid import cusgradtrades_ellipsoid_loss


def get_outputs_loss(model, optimizer, base_loss_fn, x, y, **kwargs):

    n_classes = kwargs['n_classes']
    norm = kwargs['norm']
    eps = kwargs['eps']
    device = kwargs['device']
    clip_img = kwargs['clip_img']
    normalized = kwargs['normalized']
    loss_name = kwargs['loss_name']
    glob_eps = kwargs['glob_eps']
    current_epoch = kwargs['current_epoch']
    total_epochs = kwargs['total_epochs']
    current_eps = kwargs['current_eps']

    batch_size = x.shape[0]

    clip_min, clip_max = None, None
    if clip_img:
        if normalized:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            clip_min = (np.zeros(3) - mean) / std
            clip_max = (np.ones(3) - mean) / std
        else:
            clip_min, clip_max = 0, 1

    if torch.is_tensor(eps):
        eps = eps.unsqueeze(1)

    if 'cusgradtrades' in loss_name:
        beta = parse_loss_name(loss_name, "cusgradtrades")
        steps = parse_num_steps(loss_name, 'K')

        thresh = 0.5
        if "t0.4" in loss_name:
            thresh = 0.4
        elif "t0.6" in loss_name:
            thresh = 0.6
        elif "t0.7" in loss_name:
            thresh = 0.7

        if 'adapt' in loss_name:
            thresh = thresh + thresh * (1 - current_epoch / total_epochs)

        if isinstance(eps, float):
            step_size = current_eps * 2 / steps
        elif callable(eps[0]):
            step_size = glob_eps * 2 / steps
        else:
            step_size = current_eps * 2 / steps

        if isinstance(eps, list):
            eta = glob_eps
        else:
            eta = get_cusgrad_stepsize(loss_name, eps)

        if "cusgradtrades6v3twotimesce" in loss_name:
            assert ((thresh == 0.5) and (eta == 0.1) and (beta == 6))
        elif "cusgradtrades6v3t0.6twotimesce" in loss_name:
            assert ((thresh == 0.6) and (eta == 0.1) and (beta == 6))

        outputs, loss, current_eps = cusgradtrades_loss(
            loss_name, model, base_loss_fn, x, y, norm, optimizer,
            current_eps, eta, threshold=thresh, clip_min=clip_min, clip_max=clip_max,
            step_size=step_size, epsilon=eps, perturb_steps=steps, beta=beta)
        return outputs, loss, current_eps

    elif 'trades' in loss_name:
        beta = parse_loss_name(loss_name, "trades")
        steps = parse_num_steps(loss_name, 'K')

        version = get_trades_version(loss_name)

        if isinstance(eps, float):
            step_size = eps * 2 / steps
        elif callable(eps[0]):
            step_size = glob_eps * 2 / steps
        else:
            step_size = eps * 2 / steps

        if 'gitrades' in loss_name:
            # gradually increase
            eps_i = current_epoch  / total_epochs * eps
        else:
            eps_i = eps

        outputs, loss = trades_loss(
            model, base_loss_fn, x, y,
            norm=norm, optimizer=optimizer, clip_min=clip_min, clip_max=clip_max,
            step_size=step_size, epsilon=eps_i, perturb_steps=steps, beta=beta,
            device=device, current_epoch=current_epoch, total_epochs=total_epochs,
            version=version
        )
    
    elif 'mixup' in loss_name:
        model.train()
        outputs = model(x)
        alpha = 1.0
        kld = nn.KLDivLoss(reduction='batchmean')

        if len(x) % 2 == 1:
            x = x[:len(x)-1]
            y = y[:len(y)-1]
        batch_size = y.shape[0]
        y_onehot = torch.FloatTensor(batch_size, n_classes).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(batch_size, -1), 1)

        x1, x2 = x[:len(x)//2], x[len(x)//2:]
        y1, y2 = y_onehot[:len(y)//2], y_onehot[len(y)//2:]

        lambd = np.random.beta(alpha, alpha)
        x = lambd * x1 + (1.-lambd) * x2
        yhat = y1 * lambd + (1.-lambd) * y2

        loss = kld(F.log_softmax(model(x), dim=1), yhat)

    else:
        if 'trn' in loss_name:
            model.train()
        else:
            model.eval()

        steps = parse_num_steps(loss_name, 'K')
        if isinstance(eps, float):
            step_size = eps * 2 / steps
        elif callable(eps[0]):
            step_size = glob_eps * 2 / steps
        else:
            step_size = eps * 2 / steps

        if 'adv' in loss_name:
            x = projected_gradient_descent(model, x, y=y, clip_min=clip_min,
                clip_max=clip_max, eps_iter=step_size, eps=eps, norm=norm, nb_iter=steps)
        model.train()

        optimizer.zero_grad()
        outputs = model(x)
        loss = base_loss_fn(outputs, y)
        loss = loss.mean()

    return outputs, loss
