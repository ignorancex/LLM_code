import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import qr
import numpy as np
import torchvision

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super(BasicBlock, self).__init__()
        self.conv_one = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_one  = nn.BatchNorm2d(planes)
        self.conv_two = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_two  = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn_one = self.bn_two = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = self.conv_one(x)
        out = self.bn_one(out)
        out = F.relu(out)
        out = self.conv_two(out)
        out = self.bn_two(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super(Bottleneck, self).__init__()
        self.conv_one = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn_one   = nn.BatchNorm2d(planes)
        self.conv_two = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_two   = nn.BatchNorm2d(planes)
        self.conv_three = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn_three   = nn.BatchNorm2d(self.expansion*planes)

        if not use_batchnorm:
            self.bn_one = self.bn_two = self.bn_three = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = self.conv_one(x)
        out = self.bn_one(out)
        out = F.relu(out)
        out = self.conv_two(out)
        out = self.bn_two(out)
        out = F.relu(out)
        out = self.conv_three(out)
        out = self.bn_three(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.layer_1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn    = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer_2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer_3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer_4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer_5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer_6 = nn.Linear(512*block.expansion, num_classes)
        self.num_layers = 6

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.layer_1(x)))
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.layer_6(out)
        return out

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=False):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.use_batchnorm = use_batchnorm
        self.layer_1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn    = nn.BatchNorm2d(16) if use_batchnorm else nn.Sequential()
        self.layer_2 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer_3 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer_4 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.classifier = nn.Linear(64*block.expansion, num_classes)
        # self.num_layers = 3

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden_state: bool = False):
        out = self.layer_1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        logit = self.classifier(out)
        if return_hidden_state:
            return None, logit, out
        else:
            return logit


class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10, use_batchnorm=False):
        super(WResNet_cifar, self).__init__()
        self.in_planes = 16*k
        self.use_batchnorm = use_batchnorm
        self.layer_1 = nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16*k) if use_batchnorm else nn.Sequential()
        self.layer_2 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer_3 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer_4 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        self.layer_5 = nn.Linear(64*k*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.layer_5(out)
        return out


# CIFAR-10/100 models
def ResNet20(num_classes):
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet32(num_classes):
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet44(num_classes):
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet50(num_classes):
    depth = 50
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet56(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet110(num_classes):
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)


def WRN56_2(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 2, num_classes)

def WRN56_4(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 4, num_classes)

def WRN56_8(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 8, num_classes)


# ImageNet models
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

class Pretrained_ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(Pretrained_ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.remove_batchnorm(self.model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.classifier = self.model.fc
    
    def remove_batchnorm(self, module):
        """
        Recursively remove batch normalization modules from the given module.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                self.remove_batchnorm(child)
    
    def forward(self, x, return_hidden_state = False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        out = torch.flatten(x, 1)
        logit = self.model.fc(out)

        if return_hidden_state:
            return None, logit, out
        else:
            return logit
