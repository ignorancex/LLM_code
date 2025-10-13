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
import numpy as np

class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.layer_1 = nn.Conv2d(3, 32, 3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer_2 = nn.Conv2d(32, 64, 3)
        self.layer_3 = nn.Conv2d(64, 64, 3)
        self.layer_4 = nn.Linear(64 * 4 * 4, 64)
        self.classifier =  nn.Linear(64, 100) 
        self.num_layers = 5

    def forward(self, x, return_hidden_state: bool = False):
        x = self.maxpool(F.relu(self.layer_1(x)))
        x = self.maxpool(F.relu(self.layer_2(x)))
        x = F.relu(self.layer_3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.layer_4(x))
        logit = self.classifier(x)
        if return_hidden_state:
            return None, logit, x
        else:
            return logit


class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.layer_1 = nn.Conv2d(3, 32, 3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer_2 = nn.Conv2d(32, 64, 3)
        self.layer_3 = nn.Conv2d(64, 64, 3)
        self.layer_4 = nn.Linear(64 * 4 * 4, 64)
        self.classifier =  nn.Linear(64, 10) 
        self.num_layers = 5

    def forward(self, x):
        x = self.maxpool(F.relu(self.layer_1(x)))
        x = self.maxpool(F.relu(self.layer_2(x)))
        x = F.relu(self.layer_3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.layer_4(x))
        x = self.classifier(x)
        return x

class CNNCifar10FedSam(nn.Module):
    def __init__(self):
        super(CNNCifar10FedSam, self).__init__()
        self.num_classes = 10

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, self.num_classes)
        )

        self.size = self.model_size()


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size

class CNNCifar10ForFedrod(nn.Module):
    def __init__(self):
        super(CNNCifar10ForFedrod, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3) #0
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(32, 64, 3) # 2
        self.layer3 = nn.Conv2d(64, 64, 3) #3
        self.layer4 = nn.Linear(64 * 4 * 4, 64)#4
        # proto classifier
        self.linear_proto = nn.Linear(64, 64)
        self.proto_classifier = Proto_Classifier(64, 10)
        self.scaling_train = torch.nn.Parameter(torch.tensor(10.0))

        # linear classifier
        # set bias as false only for motivation figure
        self.linear_head = nn.Linear(64, 10)

    def forward(self, x, *args, **kwargs):
        x = self.maxpool(F.relu(self.layer1(x)))
        x = self.maxpool(F.relu(self.layer2(x)))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.layer4(x))
        out = x

        # proto classifier: generate normalized features
        feature = self.linear_proto(x)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)

        # linear classifier
        logit = self.linear_head(x)

        return feature, logit, out

class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.proto = M.cuda()

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def load_proto(self, proto):
        self.proto = copy.deepcopy(proto)

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d
        return target
    
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(       #(1*28*28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),    #(16*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(16*14*14)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(16,32,5,1,2),  #32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32*7*7
        )
        self.classifier = nn.Linear(32*7*7,10)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   #(batch,32,7,7)
        x = x.view(x.size(0),-1) #(batch,32*7*7)
        feature = x
        output = self.classifier(x)
        return output