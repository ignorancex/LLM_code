import torch.nn as nn
import torch
from .blocks import ConvolutionalBlock, DeConvolutionalBlock, BottleConvolutionalBlock
from .base_net import BaseNet
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_input_channels, rep_dim, filters, use_batchnorm, activation_unit, voxel):
        super().__init__()

        # Downsampling branch
        down_channels = [num_input_channels] + filters[:-1]
        self.down_blocks = [
            ConvolutionalBlock(in_channel,out_channel,use_batchnorm, activation_unit)
            for in_channel, out_channel in zip(down_channels, down_channels[1:])
        ]
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.fc1 = nn.Linear(filters[-1]*voxel*voxel*voxel, rep_dim, bias=False)
        self.bottleneck = BottleConvolutionalBlock(filters[-2], filters[-1], use_batchnorm, activation_unit)

        for i, block in enumerate(self.down_blocks):
            self.add_module(f"down_block_{i}", block)

    def forward(self, inputs):
        # Downsampling branch
        out = inputs #torch.Size([12, 1, 64, 64, 64])
        for block in self.down_blocks:
            out = block(out)
            out = self.maxpool(out)
        # Bottleneck layer
        out = self.bottleneck(out)  #torch.Size([12, 128, 16, 16, 16])
        out = self.maxpool(out)     #torch.Size([12, 128, 8, 8, 8])
        out = out.view(out.size(0), -1)     #torch.Size([12, 65536])

        out = self.fc1(out)  #torch.Size([12, 128])
        return out


class Decoder(nn.Module):
    def __init__(self,num_output_channels, rep_dim, filters, use_batchnorm, activation_unit, voxel):
        
        super().__init__()

        self.filters = filters
        self.voxel = voxel
        self.fc2 = nn.Linear(rep_dim,self.filters[-1]*voxel*voxel*voxel, bias=False)
        
        # Upsampling branch
        up_chanels = filters[::-1]

        self.up_blocks = [
            DeConvolutionalBlock(in_channel,out_channel,use_batchnorm, activation_unit)
            for in_channel, out_channel in zip(up_chanels, up_chanels[1:])
        ]

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # Final convolution (without any concatenation with skip connections)
        self.final = nn.Conv3d(filters[0], num_output_channels, 5, bias=False, padding=2)

        for i, block in enumerate(self.up_blocks):
            self.add_module(f"up_block_{i}", block)

    def forward(self, inputs):
        out = inputs 
        out = self.fc2(out) #torch.Size([12, 65536])
        out = out.view(out.size(0), self.filters[-1], self.voxel, self.voxel, self.voxel)   #torch.Size([12, 128, 8, 8, 8])
        out = self.upsampling(out) #torch.Size([12, 128, 16, 16, 16])

        # Upsampling branch
        for block in self.up_blocks:
            out = block(out)
            out = self.upsampling(out)

        # Last convolution
        out = self.final(out)   #torch.Size([12, 1, 64, 64, 64])
        out = torch.sigmoid(out)    #torch.Size([12, 1, 64, 64, 64])
        return out

class LeNet_ELU_Autoencoder(BaseNet):

    def __init__(self, filters=None, encoder_class : object = Encoder, decoder_class : object = Decoder,**config):
        ''' 
        Constructor.
        -------
        Inputs:
            config: a ConfigParser object with the model configuration
        '''
        super().__init__(**config)
        
        # this model is a patch-based unet
        self.name = self.name + '-unet'
        self.filters = filters
        self.voxel = int(64/(2**len(filters)))
        self.rep_dim = self.rep_dim
        self.voxel = int(64/(2**len(filters)))
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        # Creating encoder and decoder
        self.encoder = encoder_class(self.num_input_channels, self.rep_dim,  self.filters, self.use_batchnorm, self.activation_unit, self.voxel)
        self.decoder = decoder_class(self.num_output_channels, self.rep_dim,  self.filters, self.use_batchnorm, self.activation_unit, self.voxel)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        z_norm = self.bn1d(z)
        z_norm_relu = F.relu(z_norm)
        x_hat = self.decoder(z_norm_relu)
        return x_hat
