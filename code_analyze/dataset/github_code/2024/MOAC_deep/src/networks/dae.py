# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from numpy import fft
import numpy as np

class DAE(nn.Module):
    ''' 
    Input: noisy y, no noise no blur x_true

    Return: estimate from y and from x, should align to x_true when training
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DAE,self).__init__()
        self.enc_x = Encoder(out_channels)
        self.gen_y = Encoder(in_channels)
        self.dec = Decoder(out_channels)

    def forward(self,y, x_clean):
        feature_x,size_x = self.enc_x(x_clean)
        feature_y,size_y = self.gen_y(y)
        out_x = self.dec(feature_x,size_x) # we donnot use GAN training method, instead, we align output y and y clean to x
        out_y = self.dec(feature_y,size_y)
        return out_y,out_x,feature_x,feature_y

class Encoder(nn.Module):
    def __init__(self,in_channels):
        super(Encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=(5,5),padding=(2,2)),
            nn.MaxPool2d(2),
        )
        self.res1 = nn.Sequential(ResBlock(64),ResBlock(64))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1)),
            nn.MaxPool2d(2),
        )
        self.res2 = nn.Sequential(ResBlock(128),ResBlock(128))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,32,kernel_size=(3,3),padding=(1,1)),
            nn.MaxPool2d(2)
        )
        self.res3 = ResBlock(32)

    def forward(self,x):
        size_record = []    # for decoder padding: encoder have n//2 operation, so the upsample might require pads
        size_record.append(x.shape)
        x = self.conv1(x)
        x = self.res1(x)
        size_record.append(x.shape)
        x = self.conv2(x)
        x = self.res2(x)
        size_record.append(x.shape)
        x = self.conv3(x)
        out = self.res3(x)
        # print(size_record)
        return out,size_record

class Decoder(nn.Module):
    def __init__(self,out_channels):
        super(Decoder,self).__init__()
        self.deconv3 = nn.ConvTranspose2d(32,128,kernel_size=(2,2),stride=(2,2))
        self.res3 = nn.Sequential(ResBlock(32))
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=(2,2),stride=(2,2))
        self.res2 = nn.Sequential(ResBlock(128),ResBlock(128))
        self.deconv1 = nn.ConvTranspose2d(64,out_channels,kernel_size=(2,2),stride=(2,2))
        self.res1 = nn.Sequential(ResBlock(64))
    
    def padding(self,x1,size):
        b,c,obj_h,obj_w = size
        diffY = obj_h - x1.shape[2]
        diffX = obj_w - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1

    def forward(self,x,size_record):
        x = self.res3(x)
        x = self.deconv3(x)
        x = self.padding(x,size_record[-1])
        x = self.res2(x)
        x = self.deconv2(x)
        x = self.padding(x,size_record[-2])
        x = self.res1(x)
        out = self.deconv1(x)
        x = self.padding(x,size_record[-3]) # usually, no padding for this one
        return out


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)  # <5>
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <6>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x
    

if __name__ == '__main__':
    input_images = torch.randn((2,4,14,256))
    input_images_c = torch.randn((2,1,14,256))
    network = DAE(4,1)
    out1,out2,_,_ = network(input_images,input_images[:,0,:,:].unsqueeze(1))
    print(out1.shape, out2.shape)