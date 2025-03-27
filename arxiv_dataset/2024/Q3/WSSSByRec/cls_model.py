import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        torch.hub.set_dir('./data')

        resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        self.resnet18 = resnet18

    def forward(self, x):
        x = self.resnet18(x)
        x = F.sigmoid(x)
        return x
    
    def loss_fn(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)
