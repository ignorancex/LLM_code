import torch.nn as nn
import torch
from .blocks import ConvolutionalBlock, DeConvolutionalBlock, BottleConvolutionalBlock
from .base_net import BaseNet
import torch.nn.functional as F

class LeNet_ELU_Autoencoder(BaseNet):

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
        self.fc2 = nn.Linear(self.rep_dim,self.filters[-1]*self.voxel*self.voxel*self.voxel, bias=False)
        
        # Bottleneck layer
        self.bottleneck = BottleConvolutionalBlock(filters[-2], filters[-1], self.use_batchnorm, self.activation_unit)
        

        # Upsampling branch
        up_chanels = filters[::-1]
        self.up_blocks = [
            DeConvolutionalBlock(in_channel,out_channel,self.use_batchnorm, self.activation_unit)
            for in_channel, out_channel in zip(up_chanels, up_chanels[1:])
        ]


        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.bn2d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        self.first_up_conv = nn.Conv3d(8, 128, 5, bias=False, padding=2)

        # Final convolution (without any concatenation with skip connections)
        self.final = nn.Conv3d(filters[0], self.num_output_channels, 5, bias=False, padding=2)

        for i, block in enumerate(self.down_blocks):
            self.add_module(f"down_block_{i}", block)
        
        for i, block in enumerate(self.up_blocks):
            self.add_module(f"up_block_{i}", block)





    def forward(self, inputs, return_encoding = False):
        # Downsampling branch
        out = inputs
        # out = torch.transpose(out, 0, 1)
        for block in self.down_blocks:
            # print('out',out.shape)
            out = block(out)
            out = self.maxpool(out)
        
        # Bottleneck layer
        out = self.bottleneck(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)

        # out = self.bn1d(self.fc1(out))
        out = self.fc1(out)
        self.encoded = out
        out = self.fc2(out)
        out = out.view(out.size(0), self.filters[-1], self.voxel, self.voxel, self.voxel)
        #out = F.elu(out)
        #out = self.first_up_conv(out)
        out = self.upsampling(out)

        # Upsampling branch
        for block in self.up_blocks:
            out = block(out)
            out = self.upsampling(out)

        # Last convolution
        out = self.final(out)
        out = torch.sigmoid(out)
        if return_encoding:
            return out,self.encoded
        return out
