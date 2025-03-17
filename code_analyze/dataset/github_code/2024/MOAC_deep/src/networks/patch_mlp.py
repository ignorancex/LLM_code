# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from numpy import fft
import numpy as np

class MLP_Net(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(MLP_Net,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_layers = nn.Sequential(
            # nn.Linear(self.in_channels*14*14,2048),   # Done in the patch slice step
            # nn.Tanh(),
            nn.Linear(2048,2048),
            nn.Tanh(),
            nn.Linear(2048,2048),
            nn.Tanh(),
            nn.Linear(2048,2048),
            nn.Tanh(),
            nn.Linear(2048,self.out_channels*14*14)            
        )
        self.patch_in = nn.Conv2d(in_channels,2048,kernel_size=(14,14),stride=(14,14))
        self.tanh = nn.Tanh()
    
    def patch_slice(self,x):
        '''
        For given image, slice into 14x14 patches, with stride=3. 
        Then linear transform them into 2048 dimensional vectors.

        (Similiar to Transformer patch embedding)
        '''
        # input Bx4xHxW
        patches = self.patch_in(x)  # Bx2048 x N1xN2
        patches = self.tanh(patches)
        b,c,h,w = patches.shape 
        self.b = b
        self.c = c
        self.n1 = h
        self.n2 = w
        patches = patches.view(b,c,-1) # Bx2048 x N1N2
        patches = patches.transpose(1,2).contiguous()  # B x N1N2 x 2048
        patches = patches.view(-1,2048) # BN1N2 x 2048
        return patches


    def patch_recover(self,x_patches):
        '''
        For given patches, replace them and add them together
        '''
        # input BN1N2 x self.out_channels*14*14
        x_patches = x_patches.view(self.b,self.n1,self.n2,-1)
        x_patches = x_patches.view(self.b,self.n1,self.n2,self.out_channels,14,14)
        x_patches = x_patches.permute(0,3,1,4,2,5).contiguous() # BxCxN1x14xN2x14'
        x_patches = x_patches.view(self.b,self.out_channels,self.n1,14,self.n2*14)
        x_out = x_patches.view(self.b,self.out_channels,self.n1*14,self.n2*14)
        return x_out


    def forward(self,x):
        ob,oc,oh,ow = x.shape
        padded_image = F.pad(x, (0, 13 - (x.shape[3]-1) % 14, 
                                 0, 13 - (x.shape[2]-1) % 14))
        x_patches = self.patch_slice(padded_image)
        x_patches = self.mlp_layers(x_patches)
        out = self.patch_recover(x_patches)
        out = out[:,:,:oh,:ow]  # crop
        return out
    

if __name__ == '__main__':
    input_images = torch.randn((2,4,14,256))
    network = MLP_Net(4,1)
    out = network(input_images)
    print(out.shape)