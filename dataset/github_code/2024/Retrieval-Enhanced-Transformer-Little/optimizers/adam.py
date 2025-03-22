import math
from typing import Dict, Any, Tuple, Optional
import torch
from labml import tracker
from torch import nn
from labml_nn.optimizers import GenericAdaptiveOptimizer, WeightDecay
class Adam(GenericAdaptiveOptimizer):
    def __init__(self, params,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 defaults: Optional[Dict[str, Any]] = None):
        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults())
        super().__init__(params, defaults, lr, betas, eps)
        self.weight_decay = weight_decay
        self.optimized_update = optimized_update
    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    def get_mv(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        beta1, beta2 = group['betas']
        m, v = state['exp_avg'], state['exp_avg_sq']
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        return m, v
    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        return group['lr']
    def adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter,m: torch.Tensor, v: torch.Tensor):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        lr = self.get_lr(state, group)
        if self.optimized_update:
            denominator = v.sqrt().add_(group['eps'])
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)
        else:
            denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = lr / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)
    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        grad = self.weight_decay(param, grad, group)
        m, v = self.get_mv(state, group, grad)
        state['step'] += 1
        self.adam_update(state, group, param, m, v)
