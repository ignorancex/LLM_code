from torch.nn import Module

class Loss(Module):
    def __init__(self, trainFactory, **args):
        super().__init__()
        self.trainFactory = trainFactory

    def __call__(self, *args):
        raise NotImplementedError
