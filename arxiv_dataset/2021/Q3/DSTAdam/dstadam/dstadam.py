import math
import torch
from torch.optim.optimizer import Optimizer, required


class DSTAdam(Optimizer):
    
    r"""Implements DSTAdam algorithm.
    base on: https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/adam.py
             https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py
    
    It has been proposed in `A decreasing scaling transition scheme from Adam to SGD`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        moment(float, optional): transition moment, moment = transition_iters / iters
        
        iters(int, required): iterations
            iters = math.ceil(trainSampleSize / batchSize) * epochs
                    
        coeff(float, optional): scaling coefficient
      
        up_lr(float, optional): upper learning rate
        low_lr(float, optional): lower learning rate 

    .. _A decreasing scaling transition scheme from Adam to SGD:
        https://arxiv.org/abs/2106.06749
    """

    def __init__(self, params, iters=required, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, coeff=1e-8, up_lr=1, low_lr=0.005):
        if not 1.0 < iters:
            raise ValueError("Invalid iters: {}".format(iters))              
        if not 0.0 < coeff < 1:
            raise ValueError("Invalid scaling coefficient: {}".format(coeff))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= up_lr:
            raise ValueError("Invalid up learning rate: {}".format(up_lr))
        if not 0.0 <= low_lr:
            raise ValueError("Invalid low learning rate: {}".format(low_lr))            
        if not low_lr <= up_lr:
            raise ValueError("required up_lr  >= low_lr, but (up_lr = {}, low_lr = {})".format(up_lr, low_lr))       
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, coeff=coeff, iters=iters, up_lr=up_lr, low_lr=low_lr)
        super(DSTAdam, self).__init__(params, defaults)
        
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))
        

    def __setstate__(self, state):
        super(DSTAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                coeff = group['coeff']
                iters = group['iters']
                rho = 10 ** (math.log(coeff, 10) / iters)
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                denom = step_size / denom
               
                #Decreasing the constant learning rate
                if not state['step'] <= iters:
                    raise ValueError("Invalid iters: {}, iters = math.ceil(train_size / batch_size) * epochs".format(iters))   
                decreasing_lr = (group['up_lr'] - group['low_lr']) * (1 - state['step'] / iters) + group['low_lr']
                #Scaling the adaptive learning rate
                denom = decreasing_lr + (denom - decreasing_lr) * (rho ** state['step'])

                # lr_scheduler cannot affect decreasing_lr, this is a workaround to apply lr decay
                decay_lr = group['lr'] / base_lr
                p.data.addcmul_(exp_avg, denom, value = -decay_lr)

        return loss