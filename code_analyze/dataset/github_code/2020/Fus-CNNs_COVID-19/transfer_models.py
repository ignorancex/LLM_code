#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:56:24 2020

@author: endiqq
"""


from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Vgg16bn_bn_conv(nn.Module):
    def __init__(self, num_class):
        super(Vgg16bn_bn_conv, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained = True)
        self.features = model.features[0:-1]
                
        self.classifier = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=256, out_channels = num_class, kernel_size=1),
                                        nn.BatchNorm2d(num_class),
                                        nn.ReLU(inplace=True))      

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.avg_pool2d(x, (x.shape[-2], x.shape[-1])).squeeze(2).squeeze(2)
        return x



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting == 'fc':
        for param in model.parameters():
            param.requires_grad = False

class Transfer_learning:
    def def_model(self, model_name, num_classes, feature_extract, use_pretrained):

        model_ft = None
        input_size = 0
    
        if model_name == "Res18" or model_name == "Res34" or model_name == "Res50":
            """ Resnet18
            """
            if model_name == "Res18":
                model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
            elif model_name == 'Res34':
                model_ft = torchvision.models.resnet34(pretrained=use_pretrained)
            else:
                model_ft = torchvision.models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "Alexnet":
            """ Alexnet
            """
            model_ft = torchvision.models.alexnet(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            set_parameter_requires_grad(model_ft, feature_extract)        
            input_size = 227
            
        elif model_name == "Vgg16bn_bn_conv":
            """ VGG16_1x1Conv
            """
            model_ft = Vgg16bn_bn_conv(num_classes)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = 512
            input_size = 224
    
        elif model_name == "Vgg16bn":
            """ VGG11
            """
            model_ft = torchvision.models.vgg16_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == "Vgg16":
            """ VGG16_nobn
            """
            model_ft = torchvision.models.vgg16(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
            
        elif model_name == "Vgg19bn":
            model_ft = torchvision.models.vgg19_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224        
        
        elif model_name == "Vgg19":
            """ VGG11_bn
            """
            model_ft = torchvision.models.vgg19(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
        elif model_name == 'xception':
            model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.last_linear.in_features
            print (num_ftrs)
            
            # model_ft.last_linear = nn.Sequential(nn.Linear(2048, 1000),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p = 0.2),
            #                                     nn.Linear(1000, 512),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p = 0.2),
            #                                     nn.Linear(512, num_classes))
    
            model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
    
            input_size = 299
               
        elif model_name == 'inceptionresnetv2':
            model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.last_linear.in_features
            # model_ft.last_linear = nn.Sequential(nn.Linear(1536, 1000),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p = 0.2),
            #                                     nn.Linear(1000, 512),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p = 0.2),
            #                                     nn.Linear(512, num_classes))
            model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
            print (num_ftrs)
            input_size = 299
            
        elif  model_name == 'efficientnet-b4':
            model_ft = EfficientNet.from_pretrained('efficientnet-b4')
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft._fc.in_features
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
            # model_ft._fc = nn.Sequential(nn.Linear(1792, 1000),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p = 0.2),
            #                             nn.Linear(1000, 512),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p = 0.2),
            #                             nn.Linear(512, num_classes))
            print(num_ftrs)
            input_size = 299        
            
        else:
            print("Invalid model name, exiting...")
            exit()
    
        return model_ft, input_size, use_pretrained, num_ftrs      