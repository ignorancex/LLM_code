import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

class Resnet3D(nn.Module):
    def __init__(self, orig_resnet3d, fc_dim=512, pool_type='maxpool', conv_size=3):
        super(Resnet3D, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(orig_resnet3d.children())[:-1])
        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def forward(self, x, pool=True):

        x = self.features(x)
        # x = self.fc(x)

        x = x.view(x.size(0), -1, 1, 1)

        return x