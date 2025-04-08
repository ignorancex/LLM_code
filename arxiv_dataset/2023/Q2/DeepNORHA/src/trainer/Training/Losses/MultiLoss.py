from .Loss import Loss


class MultiLoss(Loss):
    def __init__(self, losses, **args):
        super().__init__(**args)
        self.losses = [self.trainFactory.getLoss(loss) for loss in losses]
        for i, loss in enumerate(self.losses):
            self.add_module(f"loss_{i}", loss)

    def __call__(self, *args):
        losses = [l(*args) for l in self.losses]
        return sum(losses)

    def to(self, device):
        for l in self.losses:
            l.to(device)
