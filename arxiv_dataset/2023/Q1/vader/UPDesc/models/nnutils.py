import os
import sys
import functools
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
def get_norm(norm_type='instance_norm', dim='1d', trainable=False):
    bns = {
        '1d': nn.BatchNorm1d,
        '2d': nn.BatchNorm2d,
        '3d': nn.BatchNorm3d,
    }
    ins = {
        '1d': nn.InstanceNorm1d,
        '2d': nn.InstanceNorm2d,
        '3d': nn.InstanceNorm3d,
    }
    if norm_type == 'batch_norm':
        return functools.partial(bns[dim], affine=trainable)
    elif norm_type == 'instance_norm':
        return functools.partial(ins[dim], affine=trainable)
    else:
        raise NotImplementedError('Normalization layer - {} is not found'.format(norm_type))


def is_batchnorm(norm_layer):
    if type(norm_layer) == functools.partial:
        return norm_layer.func == nn.BatchNorm1d or norm_layer.func == nn.BatchNorm2d or norm_layer.func == nn.BatchNorm3d
    else:
        name = norm_layer.__class__.__name__
        return name == 'BatchNorm1d' or name == 'BatchNorm2d' or name == 'BatchNorm3d'


def hydra_params_to_dotdict(hparams):
    import omegaconf

    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if v is None:
                res[k] = v
            elif isinstance(v, omegaconf.DictConfig):
                res.update({k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()})
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
            elif isinstance(v, omegaconf.ListConfig):
                res[k] = omegaconf.OmegaConf.to_container(v)
            else:
                raise RuntimeError('The type of {} is not supported.'.format(type(v)))
        return res

    return _to_dot_dict(hparams)


def quat_to_axis_angle(x, eps=1e-8):
    # x: (B, *, 4), x0 + x1 * i + x2 * j + x3 * k, normalized
    assert x.shape[-1] == 4

    x = np.reshape(x, (-1, 4))
    xyz = x[:, 1:] / (np.linalg.norm(x[:, 1:], axis=1, keepdims=True) + eps)
    theta = 2.0 * np.arctan2(np.linalg.norm(x[:, 1:], axis=1, keepdims=True), x[:, :1] + eps)
    return np.concatenate((theta, xyz), axis=1)
