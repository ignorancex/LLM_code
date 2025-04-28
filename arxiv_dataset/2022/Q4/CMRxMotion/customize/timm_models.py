import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm import utils as U

class EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, input_channels=3, drop_rate=0., freeze=False):
        super().__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', 
                                            num_classes=num_classes, 
                                            pretrained=pretrained,                                              
                                            in_chans=input_channels,
                                            drop_rate=0.5)

        if freeze:
            submodules = [n for n, _ in self.efficientnet.named_children()]
            U.freeze(self.efficientnet, include_bn_running_stats=False)
            U.unfreeze(self.efficientnet, submodules[:submodules.index('blocks')])
            U.unfreeze(self.efficientnet, ['classifier'])

        pass

    def forward(self, x):
        x = self.efficientnet(x)

        return x


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, input_channels=3, freeze=False):
        super().__init__()
        self.resnet = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes, in_chans=input_channels, 
                                        norm_layer=nn.BatchNorm2d, drop_rate=0.5)
        
        if freeze:
            submodules = [n for n, _ in self.resnet.named_children()]
            U.freeze(self.resnet, include_bn_running_stats=False)
            U.unfreeze(self.resnet, submodules[:submodules.index('layer1')])
            U.unfreeze(self.resnet, ['fc'])

            self.resnet.layer2[0].conv1.weight.requires_grad
            pass
        

    def forward(self, x):
        x = self.resnet(x)
        return x
    


class ViT(nn.Module):
    def __init__(self, num_classes, pretrained=False, input_channels=3, freeze=False):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes, in_chans=input_channels)

        if freeze:
            submodules = [n for n, _ in self.vit.named_children()]
            U.freeze(self.vit, include_bn_running_stats=False)
            U.unfreeze(self.vit, submodules[:submodules.index('blocks')])
            U.unfreeze(self.vit, ['head'])

    def forward(self, x):
        x = self.vit(x)
        return x
