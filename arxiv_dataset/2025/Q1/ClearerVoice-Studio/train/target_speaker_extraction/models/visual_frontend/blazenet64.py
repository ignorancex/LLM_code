import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


EPS = 1e-8

class Visual_encoder(nn.Module):
    def __init__(self, args):
        super(Visual_encoder, self).__init__()
        self.args = args

        self.args.image_size = 128

        if self.args.causal:
            padding = (4,3,3)
        else:
            padding = (2,3,3)

        dim_3d = 8
        self.conv = nn.Conv3d(1, dim_3d, kernel_size=(5,7,7), stride=(1,1,1), padding=padding)
        self.norm = nn.BatchNorm3d(dim_3d, momentum=0.01, eps=0.001)
        self.act = nn.ReLU()
                            
        self.v_net = BlazeNet()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, batch):

        batch = F.interpolate(batch, size=(self.args.image_size,self.args.image_size), mode='bilinear', align_corners=False)

        batchsize = batch.shape[0]
        batch = self.conv(batch.unsqueeze(1))
        if self.args.causal:
            batch = batch[:,:,:-4,:,:]
        batch = self.act(self.norm(batch))

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])

        batch = self.v_net(batch)
        batch = batch.reshape(batchsize, -1, batch.shape[1]).transpose(1,2)

        return batch


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.norm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        out = self.convs(h) + x
        out = self.norm(out)
        return self.act(out)

class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeNet(nn.Module):
    def __init__(self):
        super(BlazeNet, self).__init__()
        self._define_layers()

    def _define_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24, stride=2),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )
        self.final = FinalBlazeBlock(96)
        self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

       
    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]   

        x = self.backbone(x) 
        h = self.final(x) 


        c1 = self.classifier_8(x)    
        c1 = c1.permute(0, 2, 3, 1) 
        c1 = c1.reshape(b, -1, 1)  

        c2 = self.classifier_16(h)  
        c2 = c2.permute(0, 2, 3, 1)   
        c2 = c2.reshape(b, -1, 1)   

        c = torch.cat((c1, c2), dim=1) 

        return c


