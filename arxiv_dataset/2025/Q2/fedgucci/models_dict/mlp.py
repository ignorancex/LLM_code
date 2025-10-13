import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math

# Deeper MLP for MNIST and CIFAR-10
class MLP_h1_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h1_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 10)
        self.num_layers = 2

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = self.layer_1(x)
        return x
    

class MLP_h2_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 200)
        self.layer_2 = nn.Linear(200, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class MLP_h3_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h3_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 10)
        self.num_layers = 4

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

class MLP_h4_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h4_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 200)
        self.layer_4 = nn.Linear(200, 10)
        self.num_layers = 5

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x

class MLP_h5_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h5_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 200)
        self.layer_4 = nn.Linear(200, 200)
        self.layer_5 = nn.Linear(200, 10)
        self.num_layers = 6

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = self.layer_5(x)
        return x

class MLP_h6_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h6_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 200)
        self.layer_4 = nn.Linear(200, 200)
        self.layer_5 = nn.Linear(200, 200)
        self.layer_6 = nn.Linear(200, 10)
        self.num_layers = 7

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = self.layer_6(x)
        return x
    

class MLP_h8_w200(nn.Module):
    def __init__(self, dataset):
        super(MLP_h8_w200, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 200)
        self.layer_1 = nn.Linear(200, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 200)
        self.layer_4 = nn.Linear(200, 200)
        self.layer_5 = nn.Linear(200, 200)
        self.layer_6 = nn.Linear(200, 200)
        self.layer_7 = nn.Linear(200, 200)
        self.layer_8 = nn.Linear(200, 10)
        self.num_layers = 9

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = F.relu(self.layer_6(x))
        x = F.relu(self.layer_7(x))
        x = self.layer_8(x)
        return x
    

# Wider/Narrower MLP
class MLP_h2_w400(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w400, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 400)
        self.layer_1 = nn.Linear(400, 400)
        self.layer_2 = nn.Linear(400, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class MLP_h2_w800(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w800, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 800)
        self.layer_1 = nn.Linear(800, 800)
        self.layer_2 = nn.Linear(800, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class MLP_h2_w1600(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w1600, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 1600)
        self.layer_1 = nn.Linear(1600, 1600)
        self.layer_2 = nn.Linear(1600, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class MLP_h2_w5(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w5, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 5)
        self.layer_1 = nn.Linear(5, 5)
        self.layer_2 = nn.Linear(5, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class MLP_h2_w10(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w10, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 10)
        self.layer_1 = nn.Linear(10, 10)
        self.layer_2 = nn.Linear(10, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class MLP_h2_w25(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w25, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 25)
        self.layer_1 = nn.Linear(25, 25)
        self.layer_2 = nn.Linear(25, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class MLP_h2_w50(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w50, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 50)
        self.layer_1 = nn.Linear(50, 50)
        self.layer_2 = nn.Linear(50, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class MLP_h2_w100(nn.Module):
    def __init__(self, dataset):
        super(MLP_h2_w100, self).__init__()

        if dataset == 'mnist':
            self.input_dim = 28*28
        elif dataset == 'cifar10':
            self.input_dim = 32*32*3

        self.layer_0 = nn.Linear(self.input_dim, 100)
        self.layer_1 = nn.Linear(100, 100)
        self.layer_2 = nn.Linear(100, 10)
        self.num_layers = 3

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 200))
        self.layers.append(nn.Linear(200, 200))
        self.classifier = nn.Linear(200, 10)

    def forward(self, x): # x: (batch, )
        x = x.reshape(-1, 28 * 28)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = self.classifier(x)
        return x