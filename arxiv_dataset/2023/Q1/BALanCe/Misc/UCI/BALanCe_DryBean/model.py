'''
copy from https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0/src
'''

from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

import mc_dropout

class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes, hidden_num=8, feat_num=15):
        super().__init__(num_classes)
        self.fc1 = nn.Linear(feat_num, hidden_num)
        self.dp1 = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.dp2 = mc_dropout.MCDropout()
        self.fc3 = nn.Linear(hidden_num, num_classes)

    def mc_forward_impl(self, input: Tensor):
        h = F.relu(self.dp1(self.fc1(input)))
        h = F.relu(self.dp2(self.fc2(h)))
        h = F.log_softmax(self.fc3(h), dim=1)
        return h
        
'''
class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = mc_dropout.MCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = mc_dropout.MCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input
'''

