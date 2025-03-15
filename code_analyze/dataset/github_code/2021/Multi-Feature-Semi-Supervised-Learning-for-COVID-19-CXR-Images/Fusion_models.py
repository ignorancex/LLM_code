#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 00:10:52 2020

@author: endiqq
"""


import torch
import torch.nn as nn
import os
from torchvision import datasets
import torchvision
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import torchvision
from efficientnet_pytorch import EfficientNet
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Late_Fusion_Net(nn.Module):
    def __init__(self, NN, my_model1, my_model2, num_classes):
        super(Late_Fusion_Net, self).__init__()        
        self.NN = NN
        self.num_classes = num_classes

        if self.NN == 'Res50': 
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier1 = my_model1.fc
                self.classifier2 = my_model2.fc
                
        elif self.NN == 'xception' or self.NN == 'inceptionresnetv2':
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier1 = my_model1.last_linear
                self.classifier2 = my_model2.last_linear
                
                # self.classifier3 = nn.Linear(self.embedding_dim, self.num_classes)
        
                
        elif self.NN == 'efficientnet-b4':
                self.feature1 = my_model1
                self.feature2 = my_model2
                
                self.avgpool =  my_model1._avg_pooling
                self.dropout = my_model1._dropout
                self.swish = my_model1._swish
                
                # ori_model = EfficientNet.from_pretrained('efficientnet-b4')
                self.classifier1 = my_model1._fc
                self.classifier2 = my_model2._fc
                
                # self.classifier2 = nn.Linear(self.embedding_dim, num_classes)
                      
        # else:           
        elif self.NN == 'Alexnet' or self.NN == 'Vgg16bn':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features
            
            self.avgpool = my_model1.avgpool
            
            self.classifier1 = my_model1.classifier
            self.classifier2 = my_model2.classifier
            

        elif self.NN =='Vgg16bn_bn_conv':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features            
            
            self.classifier1 = my_model1.classifier
            self.classifier2 = my_model2.classifier

                                       
    def forward(self,x,y):       
        # if (self.NN == 'Alexnet' or self.NN == 'Vgg16' or self.NN == "Vgg16bn"
        #     or self.NN == 'Vgg19' or self.NN == 'Vgg19bn'):
        if self.NN == 'efficientnet-b4':
            x1 = self.feature1.extract_features(x)            
            x2 = self.feature2.extract_features(y)
        else:   
            x1 = self.feature1(x)            
            x2 = self.feature2(y)
        
        
        # if self.mtd == 'Sum':
        if (self.NN == 'Alexnet' or self.NN == 'Vgg16' or self.NN == "Vgg16bn"
            or self.NN == 'Vgg19' or self.NN == 'Vgg19bn'):
            
            x1 = self.avgpool(x1)
            x1 = x1.view(x1.size(0),-1)
            
            x2 = self.avgpool(x2)
            x2 = x2.view(x2.size(0),-1)
                        
            x1 = self.classifier1(x1) 
            x2 = self.classifier2(x2)            
        
        elif self.NN == 'Vgg16bn_bn_conv':
            
            x1 = self.classifier1(x1)
            x1 = F.avg_pool2d(x1, (x1.shape[-2], x1.shape[-1])).squeeze(2).squeeze(2)
            x2 = self.classifier2(x2)
            x2 = F.avg_pool2d(x2, (x2.shape[-2], x2.shape[-1])).squeeze(2).squeeze(2)             
        
        elif self.NN == 'xception':
            
            x1 = F.adaptive_avg_pool2d(x1, (1, 1))
            x1 = x1.view(x1.size(0), -1)
            
            x2 = F.adaptive_avg_pool2d(x2, (1, 1))
            x2 = x2.view(x2.size(0), -1)
        
            x1 = self.classifier1(x1)
            # x1 = F.softmax(x1)
            # print (x1)                
            x2 = self.classifier2(x2)
            # x2 = F.softmax(x2)
            # print (x2)              
        elif self.NN == 'inceptionresnetv2':
            
            # x1 = F.adaptive_avg_pool2d(x1, (8, 8))
            x1 = x1.view(x1.size(0), -1)
            
            # x2 = F.adaptive_avg_pool2d(x2, (8, 8))
            x2 = x2.view(x2.size(0), -1)
            
            x1 = self.classifier1(x1)                
            x2 = self.classifier2(x2)
            
        elif self.NN == 'efficientnet-b4':
            
            x1 = self.dropout(self.avgpool(x1))
            x1 = x1.view(x1.size(0), -1)
            x1 = self.swish(self.classifier1(x1))
            
            x2 = self.dropout(self.avgpool(x2))
            x2 = x2.view(x2.size(0), -1)
            x2 = self.swish(self.classifier2(x2))
                        
        else: # Res50 

            x1 = x1.view(x1.size(0),-1)                      
            x2 = x2.view(x2.size(0),-1)

            x1 = self.classifier1(x1) 
            x2 = self.classifier2(x2)  
            
        x3 = x1+x2
            
        return x3