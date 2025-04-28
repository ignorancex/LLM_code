import itertools

import torch


class Optimizer():

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optim, max_grad_norm=-1):
        self.optimizer = optim
        self.scheduler = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def step(self):
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss, epoch)
        else:
            self.scheduler.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, loaded_dict):
        self.optimizer.load_state_dict(loaded_dict)

    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()
