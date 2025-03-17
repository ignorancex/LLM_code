#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:43:14 2020

@author: snatch
"""
from torch import nn 
import torch


class net_linear(nn.Module):
    
    def __init__(self, indim, outdim):
        super(net_linear,self).__init__()
        #self.decoder = nn.Sequential(nn.Linear(indim, outdim))
        self.decoder = nn.Sequential(nn.Linear(indim, 3), nn.ELU(), nn.Linear(3, outdim))
        
    def decode(self, z):
        return self.decoder(z)

    def forward(self,z):
        return self.decode(z), 1
    
# class net_mnist_dec(nn.Module):
    
#     def __init__(self, indim, outdim):
#         super(net_mnist_dec,self).__init__()           
#         self.fc1=nn.Linear(indim, 150)
#         self.fc2=nn.Linear(150,300)
#         self.fc3=nn.Linear(300,600)
#         self.fc4=nn.Linear(600,outdim)
#         self.elu1=nn.ELU()
#         self.elu2=nn.ELU()
#         self.elu3=nn.ELU()
        

#     def forward(self,z):
#         x=self.fc1(z)
#         x=self.elu1(x)
#         x=self.fc2(x)
#         x=self.elu2(x)
#         x=self.fc3(x)
#         x=self.elu3(x)
#         x=self.fc4(x)
# # list(net._modules.items())

#         return x , 1  

class net_mnist_dec(nn.Module):
    
    def __init__(self, indim, outdim):
        super(net_mnist_dec,self).__init__()           
        self.decoder = nn.Sequential(nn.Linear(indim, 150), 
                                        nn.ELU(),
                                      nn.Linear(150,300), 
                                        nn.ELU(), 
                                      nn.Linear(300,600),
                                        nn.ELU(), 
                                      nn.Linear(600,outdim))
      
    def decode(self, z):
        return self.decoder(z)
    def forward(self,z):
        return self.decode(z), torch.zeros_like(z)


class net_mnist_samedim(nn.Module):
    
    def __init__(self, indim, outdim):
        super(net_mnist_samedim,self).__init__()           
        self.decoder = nn.Sequential(nn.Linear(outdim, outdim), 
                                      nn.ELU(),
                                      nn.Linear(outdim,outdim), 
                                      nn.ELU(), 
                                      nn.Linear(outdim,outdim),
                                      nn.ELU(), 
                                      nn.Linear(outdim,outdim))
      
    def decode(self, z):
        return self.decoder(z)
    def forward(self,z):
        return self.decode(z) , 1  
    
class net_mnist_enc(nn.Module):    
    def __init__(self, indim, outdim):
        super(net_mnist_enc,self).__init__()           
        self.decoder = nn.Sequential(nn.Linear(outdim, 600), 
                                     nn.ELU(),
                                     nn.Linear(600,300), 
                                     nn.ELU(), 
                                     nn.Linear(300,150),
                                     nn.ELU(), 
                                     nn.Linear(150,indim))
      
    def decode(self, z):
        return self.decoder(z)
    def forward(self,z):
        return self.decode(z) , None  


class net_mnist_AE(nn.Module):
    def __init__(self, latent_dim,image_dim):
        super(net_mnist_AE, self).__init__()
        self.encoder = net_mnist_enc(latent_dim, image_dim)
        self.decoder = net_mnist_dec(latent_dim, image_dim)
        
    def forward(self, x):
        latent = self.encoder(x)[0]
        return self.decoder(latent)[0]
    
class net_cifar10_conv(nn.Module):
    
    def __init__(self, indim):
        super(net_cifar10_conv,self).__init__()
        
        # self.fc = nn.Linear(latent_dim, 4*4*256)
        self.decoder = nn.Sequential(
            nn.Linear(indim, 2*2*128),
            Reshape(-1, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 48, 4, 4] #batch*3=48
            nn.ReLU(),
 			nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
 			nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]            
        )

    def decode(self, z):
        return self.decoder(z)
    def forward(self,z):
        return self.decode(z) , None
    
class net_cifar10_conv_enc(nn.Module):
    
    def __init__(self, indim):
        super(net_cifar10_conv_enc,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            Reshape(-1, 128*2*2),
            nn.Linear(2*2*128, indim)
        )

    def forward(self,z):
        return self.encoder(z)        

class net_cifar10_conv_AE(nn.Module):
    def __init__(self, indim):
        super(net_cifar10_conv_AE,self).__init__()
        self.encoder = net_cifar10_conv_enc(indim)
        self.decoder = net_cifar10_conv(indim)
        
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)[0]

class net_cifar10_conv_elu(nn.Module):
    
    def __init__(self, indim):
        super(net_cifar10_conv_elu,self).__init__()
        
        # self.fc = nn.Linear(latent_dim, 4*4*256)
        self.decoder = nn.Sequential(
            nn.Linear(indim, 2*2*128),
            Reshape(-1, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 48, 4, 4] #batch*3=48
            nn.ELU(),
 			nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ELU(),
 			nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ELU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
        )

    def decode(self, z):
        return self.decoder(z)
    def forward(self,z):
        return self.decode(z), None   

class net_cifar10_linear(nn.Module):
    def __init__(self, indim,outdim):
        super(CIFAR10Model,self).__init__()
        self.fc1 = nn.Linear(indim, 1536)
        self.fc2 = nn.Linear(1536, 768)
        self.fc3 = nn.Linear(768, 384)
        self.fc4 = nn.Linear(384, 128)
        self.fc5 = nn.Linear(128, output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        return out


#============= Xiaoran conv net ====================
class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, bn=True):
        super(ResBlockUp, self).__init__()
        self.act = act
        if bn:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_in, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.BatchNorm2d(filters_in),
                nn.LeakyReLU(0.2, inplace=True))
    
            self.conv2_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1)
                # ,nn.BatchNorm2d(filters_out)
                )
    
    
    
            self.conv3_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.BatchNorm2d(filters_out),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_in, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.LeakyReLU(0.2, inplace=True))
    
            self.conv2_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
                )
    
    
    
            self.conv3_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.LeakyReLU(0.2, inplace=True))

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

    
class XiaoranDec_cifar(nn.Module):
    def __init__(self, latent_dim, bn=True, sigmoid=False):
        super(XiaoranDec_cifar, self).__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(latent_dim, 4*4*256)
        self.res1 = ResBlockUp(256, 128, bn=bn)
        self.res2 = ResBlockUp(128, 64, bn=bn)
        
        if bn:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True))
    

        else:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.LeakyReLU(0.2, inplace=True))


        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)
        x = self.res1(x.reshape([-1, 256, 4, 4]))
        x = self.res2(x)
        x = self.conv1_block(x)
        x = self.conv2(x)
        if self.sigmoid:  x = self.act(x)
            
        return x , None


class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, bn=True):
        super(ResBlockDown, self).__init__()
        self.act = act
        if bn:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_in, 3, stride=2, padding=1),
                # nn.BatchNorm2d(filters_in),
                nn.LeakyReLU(0.2, inplace=True))
    
            self.conv2_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1)
                # ,nn.BatchNorm2d(filters_out)
                )
    
    
            self.conv3_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=2, padding=1)
                # ,nn.BatchNorm2d(filters_out)
            )
    
        else:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_in, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True))
    
            self.conv2_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1))
    
    
            self.conv3_block = nn.Sequential(
                nn.Conv2d(filters_in, filters_out, 3, stride=2, padding=1))
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)


    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3


class XiaoranEnc_cifar(nn.Module):
    def __init__(self, latent_dim, bn=True):
        super(XiaoranEnc_cifar, self).__init__()

        if bn:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv1_block = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True))


        self.res1 = ResBlockDown(64, 64, bn=bn)
        self.res2 = ResBlockDown(64, 128, bn=bn)
        self.conv2_block = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.reshape = Reshape(-1, 4*4*256)
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, latent_dim)
        )

    def forward(self, x):
        conv1 = self.conv1_block(x)
        z = self.res1(conv1)
        z = self.res2(z)
        z = self.conv2_block(z)
        z = self.reshape(z)
        z = self.fc(z)
        return z
    
    
class XiaoranAE_cifar(nn.Module):
    def __init__(self, latent_dim, bn=True, sigmoid=False):
        super(XiaoranAE_cifar, self).__init__()
        self.encoder = XiaoranEnc_cifar(latent_dim, bn=bn)
        self.decoder = XiaoranDec_cifar(latent_dim, bn=bn, sigmoid=sigmoid)
        
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)[0]
        
