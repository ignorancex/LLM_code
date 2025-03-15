from torch import nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_layer = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool_layer = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _= x.size()
        x_avg = self.avg_pool(x).view(b, c)
        x_avg = self.avg_pool_layer(x_avg).view(b, c, 1, 1)

        x_max = self.max_pool(x).view(b, c)
        x_max = self.max_pool_layer(x_max).view(b, c, 1, 1)
        y = (x_avg + x_max) * x
        return y 
