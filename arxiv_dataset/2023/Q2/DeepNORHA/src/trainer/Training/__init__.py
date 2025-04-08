from . import Loggers, Monitors, Losses
from pytorch_lightning import loggers as plLoggers, callbacks
from torch import nn
from .ModelTrainer import ModelTrainer
from .ModelTrainerSVDD import ModelTrainerSVDD
from torch import Tensor
import torch

class DSE(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'none') -> None:
        super(DSE, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        scores = torch.sum((input - target) ** 2, dim=tuple(range(1, target.dim())))
        return torch.mean(scores)

class TrainFactory(object):
    def get(self, cls, config, ext = False):
        if ext:
            return cls(**{k:v for k,v in config.items() if k!='Type'})
        return cls(trainFactory=self, **config)

    def getMonitor(self, config):
        cls = getattr(Monitors, config['Type'])
        return self.get(cls, config)

    def getLogger(self, name, config):
        if isinstance(config, list) or isinstance(config, tuple):
            return [self.getLogger(name,c) for c in config]

        if hasattr(plLoggers, config['Type']):
            cls = getattr(plLoggers, config['Type'])
            return self.get(cls, {'name':name,**config}, True)

        cls = getattr(Loggers, config['Type'])
        return self.get(cls, {'name':name,**config})

    def getLoss(self, config):
        if config['Type']=='dulito_mse':
            return DSE()
        if hasattr(nn, config['Type']):
            cls = getattr(nn, config['Type'])
            return self.get(cls, config, True)

        cls = getattr(Losses, config['Type'])
        return self.get(cls, config)

    def getTrainer(self, model, Loss, Monitors, **config):
        losses = self.getLoss(Loss)
        monitor = self.getMonitor({'Type':'MultiMonitor', 'monitors':Monitors})
        return ModelTrainer(model, **config, loss=losses, monitor=monitor)

    def getCallbacks(self, configs):
        rcallbacks = []
        for config in configs or []:
            cls = getattr(callbacks, config['Type'])
            rcallbacks.append(self.get(cls, config, ext=True))
        return rcallbacks



class TrainFactorySVDD(object):
    def get(self, cls, config, ext = False):
        if ext:
            return cls(**{k:v for k,v in config.items() if k!='Type'})
        return cls(trainFactory=self, **config)

    def getMonitor(self, config):
        cls = getattr(Monitors, config['Type'])
        return self.get(cls, config)

    def getLogger(self, name, config):
        if isinstance(config, list) or isinstance(config, tuple):
            return [self.getLogger(name,c) for c in config]

        if hasattr(plLoggers, config['Type']):
            cls = getattr(plLoggers, config['Type'])
            return self.get(cls, {'name':name,**config}, True)

        cls = getattr(Loggers, config['Type'])
        return self.get(cls, {'name':name,**config})

    def getLoss(self, config):
        if config['Type']=='dulito_mse':
            return DSE()
        if hasattr(nn, config['Type']):
            cls = getattr(nn, config['Type'])
            return self.get(cls, config, True)

        cls = getattr(Losses, config['Type'])
        return self.get(cls, config)

    def getTrainer(self, model, Loss, Monitors, **config):
        losses = self.getLoss(Loss)
        monitor = self.getMonitor({'Type':'MultiMonitor', 'monitors':Monitors})
        return ModelTrainerSVDD(model, **config, loss=losses, monitor=monitor)

    def getCallbacks(self, configs):
        rcallbacks = []
        for config in configs or []:
            cls = getattr(callbacks, config['Type'])
            rcallbacks.append(self.get(cls, config, ext=True))
        return rcallbacks
