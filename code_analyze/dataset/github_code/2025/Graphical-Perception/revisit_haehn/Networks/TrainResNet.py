import torch.nn as nn
import torch
from . import NetUtils
from util.Config import obj2dic
from util.Config import ConfigObj
from . import Trainer
import torchvision.models.resnet as resnet
import logging

class TrainResNet(Trainer.Trainer):

    def _createNetwork(self):
        name = self.param.name
        names=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
        if name not in names:
            logging.error("Unresolved res net type %s"%name)
        logging.info("Use network %s"%name)
        num_classes = self.param.num_classes
        method = getattr(resnet,name)
        
        self.network = method(self.param.pretrained,num_classes=num_classes)
        self.lossfunc = nn.MSELoss()

    def __init__(self,param):
        ConfigObj.default(param,"pretrained",False)
        super(TrainResNet,self).__init__(param)
        