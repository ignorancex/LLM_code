# code adapted from PyTorch MNIST tutorial
# https://github.com/pytorch/examples/tree/master/mnist

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # first layer
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        # second layer
        self.conv2 = nn.Conv2d(32, 64 * 2, 3, 1)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        # fc layer 1 
        self.fc1 = nn.Linear(9216 * 2, 128 * 2)
        self.bn3 = nn.BatchNorm1d(128 * 2)
        # fc layer 2
        self.fc2 = nn.Linear(128 * 2, 10)

    def forward(self, x):
        # normalize
        x = (x - 0.1307) / 0.3801
        # first layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # second layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # pooling layer 
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # fc layer 1 
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        # fc layer 2 
        x = self.fc2(x)
        return x

