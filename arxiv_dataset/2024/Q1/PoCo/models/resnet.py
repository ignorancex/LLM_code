# import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.basenet import BaseNet
from collections import OrderedDict
import models

class ResNet(BaseNet):

    def __init__(self, args):

        super().__init__(args)


        self.layers = args.resnet_layers

        self.dropout = args.dropout

        model = getattr(torchvision.models, 'resnet{}'.format(self.layers))
        self.resnet = model(pretrained=True)


        self.feature = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2048, 1000)

        self.out = nn.Linear(1000, args.num_classes)




    def model_name(self):
        return 'Resnet-{}'.format(self.layers)

# make some changes to the end layer contrast to the original resnet
    def forward(self, x):

        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.resnet.relu(x)
        if self.dropout < 1:
            x = self.dropout(x)
        x = self.out(x)

        return x




