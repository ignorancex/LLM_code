import functools
import math

import torch


def _linear_decay(iteration, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0 - (iteration / total_iterations)
    return multiplier


def _linear_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then linear decay to 0
    """
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = 1.0 - ((iteration - warmup_iterations) / (total_iterations - warmup_iterations))
    return multiplier


def _cosine_decay(iteration, total_iterations):
    """
    decay using cosine decay to 0.0
    """
    multiplier = iteration / total_iterations
    multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _constant_warmup_cooldown(iteration, warmup_iterations, cooldown_iterations, total_iterations):
    if iteration < warmup_iterations:
        multipler = iteration / warmup_iterations
    elif warmup_iterations <= iteration < (total_iterations - cooldown_iterations):
        multipler = 1.0
    else:
        multipler = 1 - math.sqrt((iteration - (total_iterations - cooldown_iterations)) / cooldown_iterations)
    return multipler


def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def cosine_annealing(optimizer, total_steps):
    _decay_func = functools.partial(
        _cosine_decay,
        total_iterations=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def cosine_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    _decay_func = functools.partial(_cosine_decay_warmup, warmup_iterations=warmup_steps, total_iterations=total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def linear_warmup(optimizer, warmup_steps):
    _decay_func = functools.partial(_constant_warmup, warmup_iterations=warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def _constant_warmup_cooldown(iteration, warmup_iterations, cooldown_iterations, total_iterations):
    if iteration < warmup_iterations:
        multipler = iteration / warmup_iterations
    elif warmup_iterations <= iteration < (total_iterations - cooldown_iterations):
        multipler = 1.0
    else:
        multipler = 1 - math.sqrt((iteration - (total_iterations - cooldown_iterations)) / cooldown_iterations)
    return multipler


def constant_with_cooldown(optimizer, warmup_steps, cooldown_steps, total_steps):
    _decay_func = functools.partial(
        _constant_warmup_cooldown,
        warmup_iterations=warmup_steps,
        cooldown_iterations=cooldown_steps,
        total_iterations=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def linear_annealing(optimizer, total_steps):
    _decay_func = functools.partial(
        _linear_decay,
        total_iterations=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def linear_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    _decay_func = functools.partial(
        _linear_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
