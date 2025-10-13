'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from itertools import repeat
from torch.nn import Parameter
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _ntuple(n):
    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, first=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.first = first
        self.groups = groups

        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.mask = Parameter(torch.ones([self.out_channels, self.in_channels // self.groups, *self.kernel_size]), requires_grad=False)
        if first:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', bias=' + str(self.bias is not None) + ')'

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output =  F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        if self.first:
            output = self.bn1(output)
        else:
            output = self.bn2(output)

        self.output = F.relu(output)
        return output

    def prune(self, threshold, resample, reinit=False):
        #print("conv, prune weights, percentile_value: {}".format(threshold))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight)
            self.weight.data = self.weight.data * self.mask.data

    def prune_filters(self, prune_index, resample, reinit=False):
        #print("conv, prune {} filters".format(len(prune_index)))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_arr = self.weight.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[prune_index, :, :, :] = 0
        self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

    def mask_on_input_channels(self, prune_index, resample, reinit=False):
        #print("mask {} input channels".format(len(prune_index)))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_arr = self.weight.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[:, prune_index, :, :] = 0
        self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, first=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = MaskedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def resnet20(num_class=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_class)


def resnet32(num_class=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_class)


def resnet44(num_class=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_class)

def resnet50(num_class=200):
    return ResNet(BasicBlock, [8, 8, 8], num_classes=num_class)

def resnet56(num_class=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_class)

def resnet92(num_class=10):
    return ResNet(BasicBlock, [15, 15, 15], num_classes=num_class)

def resnet110(num_class=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_class)

def resnet152(num_class=10):
    return ResNet(BasicBlock, [25, 25, 25], num_classes=num_class)

def resnet200(num_class=10):
    return ResNet(BasicBlock, [33, 33, 33], num_classes=num_class)

def resnet302(num_class=10):
    return ResNet(BasicBlock, [50, 50, 50], num_classes=num_class)

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
