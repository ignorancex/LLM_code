import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
# from einops import rearrange, reduce, repeat


class DISENTANGLE_MODEL(nn.Module):
    def __init__(self, zdim, ch_num, batch_size, device, img_size=64):
        super().__init__()
        self.zdim = zdim
        self.ch_num = ch_num
        self.batch_size = batch_size
        self.device = device
        if img_size == 32:
            self.enc_size = 3
            hidden_input = 256 * self.enc_size * self.enc_size
        elif img_size == 64:
            self.enc_size = 7
            hidden_input = 256 * self.enc_size * self.enc_size
        hidden_units = 512
        
        self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 4, 1, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
        )
        self.fc_c = nn.Sequential(
                    nn.Linear(hidden_input, hidden_units),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim/2)),
                    nn.ReLU(inplace=True)
                    )
        self.fc_f = nn.Sequential(
                    nn.Linear(hidden_input, hidden_units),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim/2)),
                    nn.ReLU(inplace=True)
                    )
        self.classifier_c = nn.Sequential(
                            nn.Linear(int(zdim/2),ch_num),
#                             nn.Softmax(dim=1)
        )
        self.upsample = nn.Sequential(
            nn.Linear(zdim, hidden_input),
            nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
    
    def encode(self, x, norm=False):
        z = self.encoder(x)
        z = z.view(z.shape[0],-1)
        z_c = self.fc_c(z)
#         z_c = F.normalize(z_c)
        z_f = self.fc_f(z) 
#         z_f = F.normalize(z_f)
        output_c = self.classifier_c(z_c)
#         output_f = self.classifier_f(z_f)
        return z_c, z_f, output_c 
    
    def decode(self, z):       
        output = self.upsample(z).reshape((z.shape[0],256,self.enc_size,self.enc_size))
        output = self.decoder(output)
        # print(output.shape)
        return output
    
######################################################################################
class DISENTANGLE_MODEL_V2(nn.Module):
    def __init__(self, zdim, ch_num, font_num, batch_size,device):
        super().__init__()
        self.zdim = zdim
        self.ch_num = ch_num
        self.batch_size = batch_size
        self.device = device
        hidden_units = 512
        
        self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 4, 1, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
        )
        self.fc_c = nn.Sequential(
                    nn.Linear(256*7*7, hidden_units),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim/2)),
                    nn.ReLU(inplace=True)
                    )
        self.fc_f = nn.Sequential(
                    nn.Linear(256*7*7, hidden_units),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim/2)),
                    nn.ReLU(inplace=True)
                    )
        self.classifier_c = nn.Sequential(
                            nn.Linear(int(zdim/2),ch_num),
        )
        self.classifier_f = nn.Sequential(
                            nn.Linear(int(zdim/2),font_num),
        )
        
        self.upsample = nn.Sequential(
            nn.Linear(zdim, 256*7*7),
            nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        z = self.encoder(x)
        z = z.view(z.shape[0],-1)
        z_c = self.fc_c(z)
        z_f = self.fc_f(z)
        output_c = self.classifier_c(z_c)
        output_f = self.classifier_f(z_f)
        return z_c, z_f, output_c, output_f
    
    def decode(self, z):       
        output = self.upsample(z).reshape((z.shape[0],256,7,7))
        output = self.decoder(output)
        # print(output.shape)
        return output
