"""
https://raw.githubusercontent.com/yaodongyu/TRADES/master/trades.py
"""
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

from joblib import Parallel, delayed, parallel_backend
from ....tools.nn_utils.batch_sub_voronoi import torch_batch_project_sub_voronoi


def torch_batch_project_ellipsoid(x, tts, cents, device, step_size=0.1):
    """
    x: (n, m)
    tts: (n, keep_dims, m)
    cents: (n, m)
    """
    alpha = torch.arange(0, 1+step_size, step_size).to(device).view(-1, 1, 1)
    n_tries = alpha.shape[0]
    batch_size, fet_size = x.shape[0], x.shape[1]
    pert = (x - cents).repeat(n_tries, 1).view(n_tries, batch_size, -1)
    tts = tts.repeat(n_tries, 1, 1).view(n_tries*batch_size, tts.shape[1], fet_size)

    cents = cents.repeat(n_tries, 1).view(n_tries, batch_size, -1)
    tx = (cents + alpha * pert).view(n_tries*batch_size, fet_size, 1)
    ret = torch.bmm(tts, tx)[:, :, 0] # (n_tries*n, keep_d)
    ret = torch.le(torch.linalg.norm(ret, ord=2, dim=1), 1).view(n_tries, batch_size)
    ret = ret.float() + alpha[:, :, 0] * 1e-5
    ret = ret.argmax(dim=0)
    ret = torch.gather(tx.view(n_tries, batch_size, -1), dim=0, index=ret.view(1, -1, 1).repeat(1, 1, fet_size))[0]
    return ret


def project_delta(delta, epsilon, n_jobs=24, norm=np.inf, x_natural=None):
    if isinstance(epsilon, float):
        if norm == 2:
            avoid_zero_div = torch.tensor(1e-12, dtype=delta.dtype, device=delta.device)
            norm = torch.sqrt(torch.max(
                avoid_zero_div,
                torch.sum(delta.flatten(1) ** 2, dim=1, keepdim=True)
            ))
            factor = torch.min(
                torch.tensor(1., dtype=delta.dtype, device=delta.device),
                epsilon / norm
            )
            while len(factor.shape) < len(delta.shape):
                factor = factor.unsqueeze(-1)
            return delta * factor
        else:
            raise ValueError(f"Unsupported norm {norm}")
    elif callable(epsilon[0]):
        ori_shape = list(x_natural.shape)
        device = x_natural.get_device()
        if hasattr(epsilon[0], 'torch_batch'):
            with torch.no_grad():
                x_adv = (x_natural + delta).view(ori_shape[0], -1)
                if 'ellipsoid' in epsilon[0].torch_batch:
                    tts = torch.cat([ent.tt.to(device) for ent in epsilon], dim=0)
                    cents = torch.cat([ent.cent.to(device) for ent in epsilon], dim=0)
                    ret = torch_batch_project_ellipsoid(x_adv, tts, cents, device)
                elif "sub_voronoi" in epsilon[0].torch_batch:
                    Gs = torch.cat([ent.G.to(device) for ent in epsilon], dim=0)
                    hs = torch.cat([ent.h.to(device) for ent in epsilon], dim=0)
                    cents = torch.cat([ent.cent.to(device) for ent in epsilon], dim=0)
                    ret = torch_batch_project_sub_voronoi(x_adv, cents, Gs, hs, device)
                else:
                    ValueError(f"Unsupported {epsilon[0].torch_batch}")
                ret = ret.reshape(*ori_shape)
            return ret - x_natural
        else:
            def _helper(x, fn):
                return fn(x).reshape(-1)
            x_adv = x_natural + delta
            x_adv = x_adv.detach().cpu().numpy().astype(np.float32)
            with parallel_backend("loky", inner_max_num_threads=8):
                ret = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_helper)(x, epsilon[i]) for i, x in enumerate(x_adv))
            #ret = [_helper(x, epsilon[i]).reshape(-1) for i, x in enumerate(x_adv)]
            ret = torch.from_numpy(np.array(ret)).float().to(device)
            ret = ret.reshape(*ori_shape)
            return ret - x_natural
    else:
        if norm == 2:
            avoid_zero_div = torch.tensor(1e-12, dtype=delta.dtype, device=delta.device)
            norm = torch.sqrt(torch.max(
                avoid_zero_div,
                torch.sum(delta.flatten(1) ** 2, dim=1, keepdim=True)
            ))
            while len(norm.shape) < len(epsilon.shape):
                norm = norm.unsqueeze(-1)
            factor = torch.min(
                torch.tensor(1., dtype=epsilon.dtype, device=epsilon.device),
                epsilon / norm
            )
            while len(factor.shape) < len(delta.shape):
                factor = factor.unsqueeze(-1)
            return delta * factor
        else:
            raise ValueError(f"Unsupported norm {norm}")

def project_x_ball(x_adv, x_natural, epsilon, n_jobs=24, norm=np.inf):
    if isinstance(epsilon, float):
        if norm == np.inf:
            return torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        #elif norm == 2:
        #    delta = x_adv - x_natural
        #    delta_norm = torch.norm(delta.flatten(1), p=2, dim=1, keepdim=True)
        #    while len(delta_norm.shape) < len(delta.shape):
        #        delta_norm = delta_norm.unsqueeze(-1)
        #    return x_natural + delta / delta_norm * epsilon
        else:
            raise ValueError(f"Unsupported norm {norm}")
    elif callable(epsilon[0]):
        # serial version
        #ret = []
        #device = x_adv.get_device()
        #for i, x in enumerate(x_adv):
        #    ret.append(epsilon[i](x.cpu().numpy()))
        #ret = torch.from_numpy(np.array(ret)).float().to(device)
        #################
        ori_shape = list(x_natural.shape)
        device = x_natural.get_device()
        if hasattr(epsilon[0], 'torch_batch'):
            with torch.no_grad():
                x_adv = x_adv.view(ori_shape[0], -1)
                if 'ellipsoid' in epsilon[0].torch_batch:
                    tts = torch.cat([ent.tt.to(device) for ent in epsilon], dim=0)
                    cents = torch.cat([ent.cent.to(device) for ent in epsilon], dim=0)
                    ret = torch_batch_project_ellipsoid(x_adv, tts, cents, device)
                elif "sub_voronoi" in epsilon[0].torch_batch:
                    Gs = torch.cat([ent.G.to(device) for ent in epsilon], dim=0)
                    hs = torch.cat([ent.h.to(device) for ent in epsilon], dim=0)
                    cents = torch.cat([ent.cent.to(device) for ent in epsilon], dim=0)
                    ret = torch_batch_project_sub_voronoi(x_adv, cents, Gs, hs, device)
                else:
                    ValueError(f"Unsupported {epsilon[0].torch_batch}")
                ret = ret.reshape(*ori_shape)
            return ret - x_natural
        else:
            def _helper(x, fn):
                return fn(x).reshape(-1)
            device = x_adv.get_device()
            x_adv = x_adv.cpu().numpy().astype(np.float32)
            with parallel_backend("loky", inner_max_num_threads=8):
                ret = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_helper)(x, epsilon[i]) for i, x in enumerate(x_adv))
            #ret = [_helper(x, epsilon[i]).reshape(-1) for i, x in enumerate(x_adv)]
            ret = torch.from_numpy(np.array(ret)).float().to(device)
            ret = ret.reshape(*ori_shape)
            return ret
    else:
        if norm == np.inf:
            return torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        #elif norm == 2:
        #    delta = x_adv - x_natural
        #    delta_norm = torch.norm(delta.flatten(1), p=2, dim=1, keepdim=True)
        #    while len(delta_norm.shape) < len(delta.shape):
        #        delta_norm = delta_norm.unsqueeze(-1)
        #    return x_natural + delta / delta_norm * epsilon
        else:
            raise ValueError(f"Unsupported norm {norm}")

def trades_loss(model, loss_fn, x_natural, y, norm, optimizer,
                clip_min=None, clip_max=None, step_size=0.003, epsilon=0.031,
                perturb_steps=10, beta=1.0, reduction='none', version=None, n_jobs=12,
                current_epoch=None, total_epochs=None, device="gpu"):

    if version not in [None, 'weighted', 'hard', 'max', 'advhard', 'square',
                       'top50', 'annealtop', 'cat']:
        raise ValueError(f"[TRADES] not supported version {version}")

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')

    if torch.is_tensor(step_size):
        shape = [-1] + [1]*(len(x_natural.shape)-1)
        step_size = step_size.view(*shape)

    if torch.is_tensor(epsilon):
        shape = [-1] + [1]*(len(x_natural.shape)-1)
        epsilon = epsilon.view(*shape)

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    if norm in [np.inf]:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1)).sum(dim=1)
                if version is not None and "plus" in version:
                    loss_kl = loss_kl \
                            / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = project_x_ball(x_adv, x_natural, epsilon, n_jobs=n_jobs, norm=norm)

            if clip_min is not None and clip_max is not None:
                if isinstance(clip_min, Iterable):
                    for i in range(3):
                        x_adv[:, i, :, :] = torch.clamp(x_adv[:, i, :, :], clip_min[i], clip_max[i])
                else:
                    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif norm == 2:
        ## Setup optimizers
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        clean_output = model(x_natural).detach()

        for _ in range(perturb_steps):
            delta.requires_grad_(True)
            x_adv = x_natural + delta

            # optimize
            #optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss_kl = (-1) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(clean_output, dim=1))
                if version is not None and "plus" in version:
                    loss_kl = loss_kl \
                            / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
                loss_kl = torch.sum(loss_kl)

            dgrad = torch.autograd.grad(loss_kl, [delta])[0]
            grad_norms = dgrad.view(batch_size, -1).norm(p=2, dim=1)
            while len(grad_norms.shape) < len(dgrad.shape):
                grad_norms = grad_norms.unsqueeze(-1)
            dgrad.div_(grad_norms)
            if (grad_norms == 0).any():
                #dgrad[grad_norms == 0] = torch.randn_like(dgrad[grad_norms == 0])
                dgrad[grad_norms.flatten() == 0] = torch.randn_like(dgrad[grad_norms.flatten() == 0])
            delta = delta - step_size * dgrad

            # projection
            if clip_min is not None and clip_max is not None:
                if isinstance(clip_min, Iterable):
                    x_adv = x_natural + delta
                    for i in range(3):
                        x_adv[:, i, :, :] = torch.clamp(x_adv[:, i, :, :], clip_min[i], clip_max[i])
                    delta = x_adv - x_natural
                else:
                    delta.data.add_(x_natural)
                    delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta = project_delta(delta, epsilon, norm=2, x_natural=x_natural)

        #delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        #for _ in range(perturb_steps):
        #    delta = Variable(delta.data, requires_grad=True)
        #    optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)
        #    adv = x_natural + delta

        #    # optimize
        #    optimizer_delta.zero_grad()
        #    with torch.enable_grad():
        #        loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
        #                                   F.softmax(clean_output, dim=1)).sum()
        #    loss.backward()
        #    # renorming gradient
        #    grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
        #    delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
        #    # avoid nan or inf if gradient is 0
        #    if (grad_norms == 0).any():
        #        delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
        #    optimizer_delta.step()

        #    # projection
        #    #delta.data.add_(x_natural)
        #    #delta.data.clamp_(0, 1).sub_(x_natural)
        #    #delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        #    if clip_min is not None and clip_max is not None:
        #        delta.data.add_(x_natural)
        #        delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
        #    delta = project_delta(delta, epsilon, norm=2)

        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Not supported Norm {norm}")

    model.train()
    optimizer.zero_grad()

    if clip_min is not None and clip_max is not None:
        if isinstance(clip_min, Iterable):
            for i in range(3):
                x_adv[:, i, :, :] = torch.clamp(x_adv[:, i, :, :], clip_min[i], clip_max[i])
        else:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

    x_adv = Variable(x_adv, requires_grad=False)
    # calculate robust loss
    outputs = model(x_natural)
    if len(y.shape) == 2:
        # for smoothed ground truth
        loss_fn = torch.nn.KLDivLoss(reduction='none')
        loss_natural = loss_fn(F.log_softmax(outputs, dim=1), y).sum(1)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_natural = loss_fn(outputs, y)

    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x_natural), dim=1)).sum(dim=1)

    if version is None:
        loss = loss_natural + beta * loss_robust
    elif 'cat' in version:
        loss = loss_natural + beta * loss_robust
    elif 'square' in version:
        loss = loss_natural + beta * loss_robust.pow(2)
    elif "plus" in version:
        loss_kl = torch.sum(loss_kl, dim=1) \
                / torch.norm(torch.flatten(x_adv - x_natural, start_dim=1), p=norm, dim=1)
        loss = loss_natural + beta * loss_robust
    elif 'advhard' in version:
        w = (model(x_adv).argmax(dim=1) != y)
        loss = loss_natural + beta * batch_size / torch.sum(w) * w * loss_robust
    elif 'hard' in version:
        w = (outputs.argmax(dim=1)==y)
        loss = loss_natural + beta * batch_size / torch.sum(w) * w * loss_robust
    elif 'max' in version:
        w = (outputs.argmax(dim=1)==y)
        loss = loss_natural + beta * batch_size * (w * loss_robust).max()
    elif 'weighted' in version:
        proba = F.softmax(outputs, dim=1).gather(1, y.view(-1,1))
        loss = loss_natural + beta * batch_size * proba / torch.sum(proba) * loss_robust
    elif 'annealtop' in version:
        portion = (1. - current_epoch / total_epochs)
        topk_loss, idx = (loss_robust).topk(max(batch_size, int(batch_size * portion)))
        loss_robust = torch.zeros_like(loss_natural)
        loss_robust[idx] = topk_loss
        loss = loss_natural + beta * loss_robust
    elif 'top50' in version:
        topk_loss, idx = (loss_robust).topk(batch_size//2)
        loss_robust = torch.zeros_like(loss_natural)
        loss_robust[idx] = topk_loss
        loss = loss_natural + beta * loss_robust

    return outputs, loss
