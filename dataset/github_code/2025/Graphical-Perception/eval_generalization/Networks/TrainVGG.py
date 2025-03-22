import torch.nn as nn
import torch
from . import NetUtils
from util.Config import obj2dic
from util.Config import ConfigObj
from . import Trainer
import torchvision.models.vgg as vgg
import logging

class TrainVGG(Trainer.Trainer):

    def _createNetwork(self):
        name = self.param.name
        if name not in ["vgg11","vgg13","vgg16","vgg19"]:
            logging.error("Cannot figure out the network config %s"%name)
            logging.error("Only support vgg11, vgg13, vgg17, vgg19")
            raise Exception("Cannot figure out the network config %s"%name)
        if self.param.norm:
            name+="_bn"
        logging.info("Use network %s"%name)
        num_classes = self.param.num_classes
        method = getattr(vgg,name)
        
        self.network = method(self.param.pretrained,num_classes=num_classes)
        self.lossfunc = nn.MSELoss()

    def __init__(self,param):
        ConfigObj.default(param,"pretrained",False)
        ConfigObj.default(param,"norm",True)
        super(TrainVGG,self).__init__(param)
        