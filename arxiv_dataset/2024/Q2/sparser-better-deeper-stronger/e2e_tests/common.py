from orthogonal.tests.test import is_col_orthogonal
from models.tests.test import is_conv_orthogonal
import torch.nn as nn
import torch


base_config = {  # contains some trash
    "batch_size": 128,
    "test_batch_size": 128,
    "multiplier": 1,
    "epochs": 200,
    "lr": 0.1,
    "seed": 17,
    "optimizer": "sgd",
    "data": "cifar10",
    "model" : 'cifar_resnet_56',
    "l2": 1.0e-4,
    'bench': False,
    'scaled': False,
    'sparse': True,
    'fix': False,
    'sparse_init': 'erk',
    'growth': 'random',
    'death': 'magnitude',
    'redistribution': 'none',
    'death_rate': 0.0,
    'density': 0.05,
    'update_frequency': 100,
    'decay_schedule': 'cosine',
    'sigma_w': 1.00,
    'sigma_b': 0.0,
    "activation": 'relu',
    'momentum': 0.9,
    'valid_split': 0.5,
    'max_threads': 1,
    'global_pruning': True,
    'normalize': False,
    'adjust': False,
    'norm_type': False,
    'opt_order': False,
    'no_cuda': True,
    'manual_stop': False,
    'jaccard': False,
    'AI_iters': 100,
    'log_preprocessing': False,
}


class MockArgs:
    def __init__(self, nonzeros: str):
        for k, v in base_config.items():
            setattr(self, k, v)
        self.more_nonzeros = True if nonzeros == 'more' else False


class MockLogger:
    def log(self, _):
        pass


def verify_orthogonal(module: nn.Module):  # apply to the model
    if isinstance(module, nn.Linear):
        w = module.weight
        n, m = w.shape
        if n >= m:
            assert is_col_orthogonal(w)
        else:
            assert is_col_orthogonal(w.T)
    if isinstance(module, nn.Conv2d):
        w = module.weight
        c_out, c_in, _, _ = w.shape
        if c_out >= c_in:
            assert is_conv_orthogonal(w)
        else:
            assert is_conv_orthogonal(w.transpose(0, 1))


def verify_bias_zero(module: nn.Module):  # apply to the model
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        b = module.bias
        if b is not None:
            assert torch.sum(torch.abs(b)).item() == 0.0
