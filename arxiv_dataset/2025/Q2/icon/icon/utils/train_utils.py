"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Optional, Union, Tuple, List
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/diffusion/transformer_for_diffusion.py#L197
def get_optim_groups(
    model: nn.Module,
    weight_decay: float,
    module_whitelist: Tuple,
    module_blacklist: Tuple
) -> List:
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.startswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, module_whitelist):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, module_blacklist):
                no_decay.add(fpn)
            elif pn.endswith("pos_embed") or pn.endswith("cls_token") or pn.endswith("reg_token"):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (len(inter_params) == 0), f"Parameters {inter_params} made it into both decay/no_decay sets!"
    assert (len(param_dict.keys() - union_params) == 0), \
        f"Parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

    optim_groups = [
        {
            'params': [param_dict[pn] for pn in sorted(list(decay))],
            'weight_decay': weight_decay,
        },
        {
            'params': [param_dict[pn] for pn in sorted(list(no_decay))],
            'weight_decay': 0.0,
        }
    ]
    return optim_groups

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/lr_scheduler.py
def get_scheduler(
    optimizer: Optimizer,
    name: Union[str, SchedulerType],
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (str or SchedulerType):
            The name of the scheduler to use.
        optimizer (torch.optim.Optimizer):
            The optimizer that will be used during training.
        num_warmup_steps (int, optional):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (int, optional):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/diffusion/ema_model.py
class EMA:
    
    def __init__(
        self,
        model: nn.Module,
        update_after_step: Optional[int] = 0,
        inv_gamma: Optional[float] = 1.0,
        power: Optional[float] = 2/3,
        min_value: Optional[float] = 0.,
        max_value: Optional[float] = 0.9999
    ) -> None:
        """
        Exponential Moving Average of models weights

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        self.optimization_step += 1