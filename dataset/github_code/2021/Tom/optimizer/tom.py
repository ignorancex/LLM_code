import torch
from torch.optim.optimizer import Optimizer
import math

class Tom(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99,0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, use_bias_correction_for_level_trend=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, use_bias_correction_for_level_trend=use_bias_correction_for_level_trend)
        super(Tom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Tom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
	
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Tom does not support sparse gradients')
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_trend_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['previous_grad'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, exp_trend_avg= state['exp_avg'], state['exp_avg_sq'], state['exp_trend_avg']
                beta1, beta2, beta3 = group['betas']
                state['step'] += 1
                if group['use_bias_correction_for_level_trend']:
                    bias_corr1 = 1 - ((beta1 * beta2) ** state['step']) 
                else:
                    bias_corr1 = 1
                bias_corr3 = 1 - (beta3 ** state['step'])
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                exp_avg.add_(exp_trend_avg).mul_(beta1).add_(grad, alpha=1-beta1)
                exp_trend_avg.mul_(beta2).add_(grad - state['previous_grad'], alpha=1-beta2) 
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1-beta3)
                total = torch.add(exp_avg, exp_trend_avg) / bias_corr1
                total = total/(exp_avg_sq.sqrt()/math.sqrt(bias_corr3)).add_(group['eps'])
                state['previous_grad'] = torch.clone(grad).detach()
                step_size = group['lr']
                p.data.add_(total, alpha = -step_size)
        return loss
