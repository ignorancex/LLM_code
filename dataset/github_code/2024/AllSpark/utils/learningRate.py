import math
import numpy as np

def adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg):
    if cur_iters < warmup_iters:
        lr = optimizer_cfg['lr'] * cur_iters / warmup_iters
    elif optimizer_cfg['lr_schedule'] == 'cosine':
        lr = optimizer_cfg["min_lr"] + (optimizer_cfg['lr'] - optimizer_cfg['min_lr']) * 0.5 * (1. + math.cos(math.pi * (cur_iters - warmup_iters) / (max_iters - warmup_iters)))
    elif optimizer_cfg['lr_schedule'] == 'poly':
        lr = max(optimizer_cfg['lr'] * math.pow((1 - (cur_iters - warmup_iters) / (max_iters - warmup_iters)), 1), optimizer_cfg['min_lr'])
    elif optimizer_cfg['lr_schedule'] == 'cycle':
        step = 1415
        cycle = np.floor(1+cur_iters/(2*step))
        x = np.abs(cur_iters/step - 2*cycle + 1)
        lr = 1e-3 + (1e-2 - 1e-3) * np.maximum(0, (1-x)) * 1/(2.**(x-1))
        
    else:
        raise NotImplementedError
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
