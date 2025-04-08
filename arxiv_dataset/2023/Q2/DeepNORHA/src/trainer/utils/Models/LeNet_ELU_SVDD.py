import torch.nn as nn
import torch
from .blocks import ConvolutionalBlock, DeConvolutionalBlock, BottleConvolutionalBlock
from .base_net import BaseNet

class LeNet_ELU_SVDD(BaseNet):

    def __init__(self, filters=None, **config):
        ''' 
        Constructor.
        -------
        Inputs:
            config: a ConfigParser object with the model configuration
        '''
        super().__init__(**config)
        
        # this model is a patch-based unet
        self.name = self.name + '-unet'
        # number of channels of each conv layer
        self.filters = filters
        self.voxel = int(64/(2**len(filters)))
        self.rep_dim = self.rep_dim
        self.encoded = None

        
        # Downsampling branch
        down_channels = [self.num_input_channels] + filters[:-1]
        self.down_blocks = [
            ConvolutionalBlock(in_channel,out_channel,self.use_batchnorm, self.activation_unit)
            for in_channel, out_channel in zip(down_channels, down_channels[1:])
        ]
        
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        
        self.fc1 = nn.Linear(self.filters[-1]*self.voxel*self.voxel*self.voxel, self.rep_dim, bias=False)

        # Bottleneck layer
        self.bottleneck = BottleConvolutionalBlock(filters[-2], filters[-1], self.use_batchnorm, self.activation_unit)
        

        for i, block in enumerate(self.down_blocks):
            self.add_module(f"down_block_{i}", block)
        

    def forward(self, inputs):
        # Downsampling branch
        out = inputs
        for block in self.down_blocks:
            out = block(out)
            out = self.maxpool(out)
        
        # Bottleneck layer
        out = self.bottleneck(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)

        # out = self.bn1d(self.fc1(out))
        out = self.fc1(out)
        return out
