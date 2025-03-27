class StepLR:
    def __init__(self, init_lr, step_size, gamma):
        self.init_lr = init_lr
        self.step_size = step_size
        self.gamma = gamma

        self.reset()

    def reset(self):
        self.lr = self.init_lr
        self.last_step = 0

    def step(self):
        self.last_step += 1
        self.lr = self.init_lr * self.gamma ** (self.last_step // self.step_size)

    def get_lr(self):
        return self.lr


__scheduler_zoo__ = {
    "step_lr": StepLR,
}


def get_scheduler(name: str, **kwargs):
    return __scheduler_zoo__[name](**kwargs)
