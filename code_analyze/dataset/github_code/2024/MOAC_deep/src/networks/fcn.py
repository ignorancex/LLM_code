# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from numpy import fft
import numpy as np

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(FCN,self).__init__()
        self.dilation = [1,2,3,4,3,2,1]
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=(3,3),padding=(1,1),padding_mode='circular'),
            nn.ReLU()
        )
        self.mid_layer = nn.ModuleList([self._layer_unit(i) for i in range(1,6)])
        self.out_layer = nn.Sequential(
            nn.Conv2d(64,4,kernel_size=(3,3),padding=(1,1),padding_mode='circular'),
        )
        self.mix_layer = nn.Conv2d(4,out_channels,kernel_size=(1,1))

    def forward(self,x):
        n = self.input_layer(x)
        for sub_mid_layer in self.mid_layer:
            n = sub_mid_layer(n)
        n = self.out_layer(n)   # residual
        out = x-n
        out = self.mix_layer(out)
        return out
    
    def _layer_unit(self,layer_id):
        return nn.Sequential(
            nn.Conv2d(64,64,
                      kernel_size=(self.dilation[layer_id]*2+1,self.dilation[layer_id]*2+1),
                      padding=(self.dilation[layer_id],self.dilation[layer_id]),
                      padding_mode='circular'
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
if __name__ == '__main__':
    input_images = torch.randn((2,4,14,256))
    network = FCN(4,1)
    out = network(input_images)
    print(out.shape)