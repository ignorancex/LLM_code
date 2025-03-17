import math
from typing import Dict
from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad
class AdamWarmupCosineDecay(AMSGrad):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,weight_decay: WeightDecay = WeightDecay(),optimized_update: bool = True,amsgrad=False, warmup=0, total_steps=1e10, defaults=None):
        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup, total_steps=total_steps))
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)
    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        if group['warmup'] > state['step']:
            return 1e-8 + state['step'] * group['lr'] / group['warmup']
        else:
            progress = (state['step'] - group['warmup']) / max(1, group['total_steps'] - group['warmup'])
            return group['lr'] * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
if __name__ == '__main__':
   pass