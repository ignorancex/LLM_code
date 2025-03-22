import sys
import os
import pickle
import warnings

import torch
import torch.nn as nn
import torchvision

import numpy as np

warnings.filterwarnings('ignore')
sys.path.append("../helpers")
sys.path.append("../models")

from vgg import VGG, make_layers
from resnet import ResNet, BasicBlock, Bottleneck
from googlenet import GoogleNet
from mobilenet_v2 import MobileNetV2 
from densenet import DenseNet, BottleneckDense


def resize_out_features(model, classes):
    if hasattr(model, "linear"):
        in_features = model.linear[-1].in_features
        new_linear = nn.Linear(in_features, out_features=classes, bias=True)
        model.linear[-1] = new_linear
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        new_linear = nn.Linear(in_features, out_features=classes, bias=True)
        model.classifier[-1] = new_linear


def freeze_layers(model, unfreeze_layer_list):
    for name, param in model.named_parameters():
        if not any(unfreeze_layers in name for unfreeze_layers in unfreeze_layer_list):
            param.requires_grad = False
