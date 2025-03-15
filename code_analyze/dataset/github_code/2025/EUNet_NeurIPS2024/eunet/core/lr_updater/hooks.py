from torch._six import inf
from mmcv.runner.hooks import HOOKS, Hook, LrUpdaterHook
from typing import Union, List, Optional
from mmcv import runner


@HOOKS.register_module()
class HoodLrUpdaterHook(LrUpdaterHook):

    def __init__(self, 
                 decay_rate: float = 1e-1,
                 decay_min: float = 0.0,
                 decay_steps: int = 5e6,
                 step_start: int = 0,
                 **kwargs) -> None:
        self.decay_rate = decay_rate
        self.decay_min = decay_min
        self.decay_steps = decay_steps
        self.step_start = step_start

        super(HoodLrUpdaterHook, self).__init__(**kwargs)
    
    def _sched_fun(self, step):
        cur_step = max(0, step - self.step_start)
        decay = self.decay_rate ** (cur_step // self.decay_steps) + 1e-2
        decay = max(decay, self.decay_min)
        return decay
    
    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        decay = self._sched_fun(progress)
        lr = base_lr * decay
        return lr