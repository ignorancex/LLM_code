import torch
from torch.optim import Optimizer
class CustomAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 v0_init='constant', v0_value=1e-6, k=5):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        v0_init=v0_init, v0_value=v0_value, k=k)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            v0_init = group['v0_init']
            v0_value = group['v0_value']
            k = group['k']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize first moment vector
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Initialize second moment vector
                    if v0_init == 'hessian':
                        # Approximate Hessian (not practical, placeholder)
                        state['exp_avg_sq'] = torch.ones_like(p.data) * v0_value
                    elif v0_init == 'grad_square':
                        state['exp_avg_sq'] = grad.pow(2)
                    elif v0_init == 'grad_variance':
                        state['grad_samples'] = [grad.clone()]
                        state['exp_avg_sq'] = torch.ones_like(p.data) * v0_value
                    elif v0_init == 'constant':
                        state['exp_avg_sq'] = torch.ones_like(p.data) * v0_value
                    elif v0_init == 'adaptive':
                        state['exp_avg_sq'] = grad.abs()
                    else:
                        raise ValueError('Invalid v0_init option')

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                if v0_init == 'grad_variance' and state['step'] < k:
                    state['grad_samples'].append(grad.clone())
                    if len(state['grad_samples']) > k:
                        state['grad_samples'].pop(0)
                    variance = torch.var(torch.stack(state['grad_samples']), dim=0, unbiased=False)
                    exp_avg_sq.copy_(variance + group['eps'])
                elif v0_init == 'grad_variance' and state['step'] == k:
                    variance = torch.var(torch.stack(state['grad_samples']), dim=0, unbiased=False)
                    exp_avg_sq.copy_(variance + group['eps'])

                state['step'] += 1

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if v0_init != 'grad_variance' or state['step'] > k:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
