from typing import Dict
from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad
class Noam(AMSGrad):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,weight_decay: WeightDecay = WeightDecay(),optimized_update: bool = True,amsgrad=False,warmup=0, d_model=512, defaults=None):
        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup))
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)
        self.d_model = d_model
    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        factor = min(state['step'] ** (-0.5), state['step'] * group['warmup'] ** (-1.5))
        return group['lr'] * self.d_model ** (-0.5) * factor
if __name__ == '__main__':
    pass