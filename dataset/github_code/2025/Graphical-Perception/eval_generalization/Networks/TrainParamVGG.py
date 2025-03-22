import torch.nn as nn
import torch
from . import NetUtils
from util.Config import obj2dic
from util.Config import ConfigObj
from . import Trainer
import torchvision.models.vgg as vgg
import logging

#copy from torchvision 0.8 

def generateActiviationLayer(name,param):
    if name=="ReLU":
        return nn.ReLU(True)
    elif name=="LeakyReLU":
        return nn.LeakyReLU(param,True)
    elif name=="PReLU":
        return nn.PReLU(init=param)
    elif name=="ELU":
        return nn.ELU(param)
    elif name=="CELU":
        return nn.CELU(param)
    elif name=="SELU":
        return nn.SELU(True)
    elif name=="GELU":
        return nn.GELU()
    else:
        logging.warning("Unknown activiation layer %s, Use ReLU instead"%name)
        return nn.ReLU()

class VGGCustom(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, activiationFunction = "relu", activiationFuctionParam = 0, dropoutRate=0.5):
        super(VGGCustom, self).__init__()
        self.features = features
        self.activiationFunction = activiationFunction
        self.activiationFunctionParam = activiationFuctionParam
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            self.genActLayer(),
            nn.Dropout(dropoutRate),
            nn.Linear(4096, 4096),
            self.genActLayer(),
            nn.Dropout(dropoutRate),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def genActLayer(self):
        return generateActiviationLayer(self.activiationFunction, self.activiationFunctionParam)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def make_layers(cfg, batch_norm=False, activiationLayer="ReLU", activiationLayerParam=0):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), generateActiviationLayer(activiationLayer, activiationLayerParam)]
            else:
                layers += [conv2d, generateActiviationLayer(activiationLayer, activiationLayerParam)]
            in_channels = v
    return nn.Sequential(*layers)



class TrainParamVGG(Trainer.Trainer):

    def _createNetwork(self):
        name = self.param.name
        tag="A"
        if name not in ["vgg11","vgg13","vgg16","vgg19"]:
            logging.error("Cannot figure out the network config %s"%name)
            logging.error("Only support vgg11, vgg13, vgg16, vgg19")
            raise Exception("Cannot figure out the network config %s"%name)
        else:
            dic={
                "vgg11":"A",
                "vgg13":"B",
                "vgg16":"D",
                "vgg19":"E"
                }
            tag = dic[name]
        
        logging.info("Use network %s"%name)
        num_classes = self.param.num_classes
        cfgs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        self.network = VGGCustom(make_layers(cfgs[tag],self.param.norm,self.param.activiationFunction,self.param.activiationFunctionParam),num_classes,True,self.param.activiationFunction,self.param.activiationFunctionParam,self.param.dropout)
        self.lossfunc = nn.MSELoss()

    def __init__(self,param):
        ConfigObj.default(param,"activiationFunction","ReLU")
        ConfigObj.default(param,"activiationFunctionParam","0")
        ConfigObj.default(param,"dropout","0.5")
        ConfigObj.default(param,"norm",False)
        super(TrainParamVGG,self).__init__(param)
        