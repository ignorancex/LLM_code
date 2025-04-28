import torch
import torch.nn as nn
import timm
import math

def create_model(args, device):
    print("Creating model")
    if args.model == "resnet10t":
        my_model = timm.create_model('resnet10t', num_classes=args.num_classes, pretrained=False)
    elif args.model == "resnet18":
        my_model = timm.create_model('resnet18', num_classes=args.num_classes, pretrained=False)
    elif args.model == "resnet50":
        my_model = timm.create_model('resnet50', num_classes=args.num_classes, pretrained=False)
        
    my_model.to(device)
    return my_model