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
           'wide_resnet50_2', 'wide_resnet101_2', 'resnet152_layer1']
        if name not in names:
            logging.error("Unresolved res net type %s"%name)
        logging.info("Use network %s"%name)
        num_classes = self.param.num_classes
        true_name=''
        if name=='resnet152_layer1':
            true_name='resnet152_layer1'
            name='resnet152'
        if name=='resnet152_mask':
            true_name='resnet152_mask'
            name='resnet152'
        if name=='resnet50_layer1':
            true_name='resnet50_layer1'
            name='resnet50'
        if name=='resnet50_mask':
            true_name='resnet50_mask'
            name='resnet50'
        method = getattr(resnet,name)
        
        self.network = method(self.param.pretrained,num_classes=num_classes)
        self.lossfunc = nn.MSELoss()
        if true_name=='resnet152_layer1':
            new_model = nn.Sequential(*list(method(self.param.pretrained,num_classes=num_classes).children())[:5],nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(in_features=256, out_features=1, bias=True))
            self.network = new_model
        if true_name=='resnet152_mask':
            self.network.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # print(self.network)
            pass
        if true_name=='resnet50_mask':
            self.network.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # print(self.network)
            pass
        # print(self.lossfunc)
        # new_model = nn.Sequential(*list(method(self.param.pretrained,num_classes=num_classes).children())[:5],nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(in_features=256, out_features=1, bias=True))
        # self.network = new_model
        # print(new_model)

    def __init__(self,param):
        ConfigObj.default(param,"pretrained",False)
        super(TrainResNet,self).__init__(param)
        