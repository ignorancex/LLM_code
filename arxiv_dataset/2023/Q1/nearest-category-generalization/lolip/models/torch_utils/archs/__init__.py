import types
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from .densenet import densenet161
from .wideresnet import *
from .resnet import resnet18, resnet50, resnet152, resnet101, resnext101_32x8d
from .resnet import ResNet50Layer3
from .vgg import Vgg19, Vgg16, Vgg13
from .mlps import VggMLP
from .inception import inception_v3
from .mobilenet import mobilenet_v3_large


def tvgg16(n_classes, n_channels):
    return vgg16(pretrained=False, num_classes=n_classes)

####################
#### MobileNet #####
####################

def MobileNetV3Large(n_classes, n_channels):
    model = mobilenet_v3_large(pretrained=False, num_classes=n_classes)
    return model

def MobileNetV3LargeNorm01(n_classes, n_channels):
    model = mobilenet_v3_large(pretrained=False, num_classes=n_classes,
                               normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return model

####################
##### DenseNet #####
####################

def DenseNet161(n_classes, n_channels):
    resnet = densenet161(pretrained=False, num_classes=n_classes)
    return resnet

def DenseNet161Norm01(n_classes, n_channels):
    resnet = densenet161(pretrained=False, num_classes=n_classes,
                         normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return resnet

####################
### Inceptionv3 ####
####################

def Inceptionv3(n_classes, n_channels):
    model = inception_v3(pretrained=False, num_classes=n_classes)
    return model

####################
###### ResNet ######
####################

def ResNet18(n_classes, n_channels):
    resnet = resnet18(pretrained=False, n_channels=n_channels, num_classes=n_classes)
    return resnet

def ResNet18Norm01(n_classes, n_channels):
    resnet = resnet18(pretrained=False, n_channels=n_channels, num_classes=n_classes,
                      normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return resnet

def preResNet18Norm01(n_classes, n_channels):
    resnet = resnet18(pretrained=True, n_channels=n_channels, num_classes=n_classes,
                      normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return resnet

def ResNet101(n_classes, n_channels):
    resnet = resnet101(pretrained=False, n_channels=n_channels, num_classes=n_classes)
    return resnet

def preResNet50(n_classes, n_channels):
    resnet = resnet50(pretrained=True, n_channels=n_channels, num_classes=n_classes)
    return resnet

def ResNet50(n_classes, n_channels):
    resnet = resnet50(pretrained=False, n_channels=n_channels, num_classes=n_classes)
    return resnet

def ResNet152(n_classes, n_channels):
    resnet = resnet152(pretrained=False, n_channels=n_channels, num_classes=n_classes)
    return resnet

def ResNet152Norm01(n_classes, n_channels):
    resnet = resnet152(pretrained=False, n_channels=n_channels, num_classes=n_classes,
                      normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return resnet

def preResNeXt101(n_classes, n_channels):
    resnet = resnext101_32x8d(pretrained=True, n_channels=n_channels, num_classes=n_classes)
    return resnet

def preResNet50Norm01(n_classes, n_channels):
    resnet = resnet50(pretrained=True, n_channels=n_channels, num_classes=n_classes,
                      normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return resnet

def ResNet50Norm01(n_classes, n_channels):
    resnet = resnet50(pretrained=False, n_channels=n_channels, num_classes=n_classes,
                      normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
    return resnet

def ResNet50Norm02(n_classes, n_channels):
    resnet = resnet50(pretrained=False, n_channels=n_channels, num_classes=n_classes,
                      normalize_mean=(0.4914, 0.4822, 0.4465), normalize_std=(0.2023, 0.1994, 0.2010))
    return resnet

class CNN001(nn.Module):
    def __init__(self, n_features, n_classes, n_channels=None, save_intermediates=False):
        super(CNN001, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.save_intermediates = save_intermediates
        self.intermediates = []

    def forward(self, x):
        if self.save_intermediates:
            del self.intermediates
            self.intermediates = []
            x = self.conv1(x)
            self.intermediates.append(x)
            x = F.relu(x)
            x = self.conv2(x)
            self.intermediates.append(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            self.intermediates.append(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            self.intermediates.append(x)
            return x
        else:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            #output = F.log_softmax(x, dim=1)
            return x

class STNCNN001(CNN001):
    def __init__(self, n_features, n_classes, n_channels=None):
        super(STNCNN001, self).__init__(n_classes=n_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return super(STNCNN001, self).forward(x)

class CNN001Init1(CNN001):
    def __init__(self, n_classes, n_channels=None):
        super(CNN001Init1, self).__init__(n_classes, n_channels=None)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

class CNN002Conv2(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_features, n_classes, drop=0.5, n_channels=1):
        super(CNN002Conv2, self).__init__()

        self.num_channels = n_channels

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, n_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def intermediate_forward(self, x, layer_index):
        _, out_list = self.intermediate_outputs(x)
        return out_list[layer_index]

    def get_repr(self, x, mode="cnn_fet"):
        if mode == "cnn_fet":
            features = self.feature_extractor(x)
            return features.view(-1, 64 * 4 * 4)
        elif mode == "last":
            x = self.feature_extractor(x)
            x = self.classifier.fc1(x)
            x = self.classifier.relu1(x)
            x = self.classifier.drop(x)
            x = self.classifier.fc2(x)
            x = self.classifier.relu2(x)
            return features.view(-1, 64 * 4 * 4)
        else:
            raise ValueError()

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class CNN002(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_features=None, n_classes=10, drop=0.5, n_channels=1, save_intermediates=False):
        super(CNN002, self).__init__()

        self.num_channels = n_channels

        self.save_intermediates = save_intermediates
        self.intermediates = []

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, n_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def intermediate_outputs(self, x):
        out_list = []

        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.relu1(x)
        out_list.append(x)
        x = self.feature_extractor.conv2(x)
        x = self.feature_extractor.relu2(x)
        x = self.feature_extractor.maxpool1(x)
        out_list.append(x)
        x = self.feature_extractor.conv3(x)
        x = self.feature_extractor.relu3(x)
        out_list.append(x)
        x = self.feature_extractor.conv4(x)
        x = self.feature_extractor.relu4(x)
        x = self.feature_extractor.maxpool2(x)
        out_list.append(x)
        x = x.view(-1, 64 * 4 * 4)

        x = self.classifier.fc1(x)
        x = self.classifier.relu1(x)
        out_list.append(x)
        x = self.classifier.drop(x)
        x = self.classifier.fc2(x)
        x = self.classifier.relu2(x)
        out_list.append(x)
        x = self.classifier.fc3(x)
        return x, out_list

    def intermediate_forward(self, x, layer_index):
        _, out_list = self.intermediate_outputs(x)
        return out_list[layer_index]

    def get_repr(self, x, mode="cnn_fet"):
        if mode == "cnn_fet":
            features = self.feature_extractor(x)
            return features.view(-1, 64 * 4 * 4)
        elif mode == "last":
            x = self.feature_extractor(x)
            x = x.view(-1, 64 * 4 * 4)
            x = self.classifier.fc1(x)
            x = self.classifier.relu1(x)
            x = self.classifier.drop(x)
            x = self.classifier.fc2(x)
            x = self.classifier.relu2(x)
            return x
        else:
            raise ValueError()

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

class CNN002uni(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_classes, drop=0.5, n_channels=1):
        super(CNN002uni, self).__init__()

        self.num_channels = n_channels

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, n_classes)),
        ]))

    def get_repr(self, x):
        features = self.feature_extractor(x)
        return features.view(-1, 64 * 4 * 4)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

class CNN003(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_features, n_classes, n_channels=1, save_intermediates=False):
        super(CNN003, self).__init__()

        self.num_channels = n_channels

        self.save_intermediates = save_intermediates
        self.intermediates = []

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 5, padding=2)),
            ('relu1', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(32, 64, 5, padding=2)),
            ('relu2', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 7 * 7, 1024)),
            ('relu3', activ),
            ('fc2', nn.Linear(1024, n_classes)),
        ]))

    def get_repr(self, x, mode="cnn_fet"):
        if mode == "cnn_fet":
            features = self.feature_extractor(x)
            return features.view(-1, 64 * 7 * 7)
        elif mode == "last":
            x = self.feature_extractor(x)
            x = x.view(-1, 64 * 4 * 4)
            x = self.classifier.fc1(x)
            x = self.classifier.relu1(x)
            return x
        else:
            raise ValueError()

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 7 * 7))
        return logits

class MLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x

class MLPv2(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(MLPv2, self).__init__()
        self.hidden = nn.Linear(n_features[0], 2048)
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x

class MLPv3(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(MLPv3, self).__init__()
        self.hidden = nn.Linear(n_features[0], 4096)
        self.fc = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x

class LargeMLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLP, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

    def intermediate_outputs(self, x):
        out_list = []

        x = F.relu(self.hidden(x))
        out_list.append(x)
        x = F.relu(self.hidden2(x))
        out_list.append(x)
        x = F.relu(self.hidden3(x))
        out_list.append(x)
        x = self.fc(x)
        return x, out_list

    def intermediate_forward(self, x, layer_index):
        x = F.relu(self.hidden(x))
        if layer_index == 1:
            x = F.relu(self.hidden2(x))
        elif layer_index == 2:
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
        return x

class LargeMLPv2(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv2, self).__init__()
        self.hidden = nn.Linear(n_features[0], 384)
        self.hidden2 = nn.Linear(384, 384)
        self.hidden3 = nn.Linear(384, 384)
        self.fc = nn.Linear(384, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv3(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv3, self).__init__()
        self.hidden = nn.Linear(n_features[0], 512)
        self.hidden2 = nn.Linear(512, 512)
        self.hidden3 = nn.Linear(512, 512)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

    def intermediate_outputs(self, x):
        out_list = []

        x = F.relu(self.hidden(x))
        out_list.append(x)
        x = F.relu(self.hidden2(x))
        out_list.append(x)
        x = F.relu(self.hidden3(x))
        out_list.append(x)
        x = self.fc(x)
        return x, out_list

    def intermediate_forward(self, x, layer_index):
        x = F.relu(self.hidden(x))
        if layer_index == 1:
            x = F.relu(self.hidden2(x))
        elif layer_index == 2:
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
        return x

class LargeMLPv4(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv4, self).__init__()
        self.hidden = nn.Linear(n_features[0], 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv5(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv5, self).__init__()
        self.hidden = nn.Linear(n_features[0], 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.fc = nn.Linear(1024, n_classes)

        nn.init.kaiming_normal_(self.hidden.weight)
        nn.init.kaiming_normal_(self.hidden2.weight)
        nn.init.kaiming_normal_(self.hidden3.weight)
        nn.init.constant_(self.hidden.bias, 0)
        nn.init.constant_(self.hidden2.bias, 0)
        nn.init.constant_(self.hidden3.bias, 0)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv6(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv6, self).__init__()
        self.hidden = nn.Linear(n_features[0], 4096)
        self.hidden2 = nn.Linear(4096, 4096)
        self.hidden3 = nn.Linear(4096, 4096)
        self.fc = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv7(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv7, self).__init__()
        self.hidden = nn.Linear(n_features[0], 4096)
        self.hidden2 = nn.Linear(4096, 4096)
        self.hidden3 = nn.Linear(4096, 4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv8(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv8, self).__init__()
        self.hidden = nn.Linear(n_features[0], 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.hidden4 = nn.Linear(1024, 1024)
        self.hidden5 = nn.Linear(1024, 1024)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.fc(x)
        return x

class LargeMLPv9(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv9, self).__init__()
        self.hidden = nn.Linear(n_features[0], 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.hidden4 = nn.Linear(1024, 1024)
        self.hidden5 = nn.Linear(1024, 1024)
        self.fc = nn.Linear(1024, n_classes)


        nn.init.kaiming_normal_(self.hidden.weight)
        nn.init.kaiming_normal_(self.hidden2.weight)
        nn.init.kaiming_normal_(self.hidden3.weight)
        nn.init.kaiming_normal_(self.hidden4.weight)
        nn.init.kaiming_normal_(self.hidden5.weight)
        nn.init.kaiming_normal_(self.fc.weight)

        nn.init.constant_(self.hidden.bias, 0)
        nn.init.constant_(self.hidden2.bias, 0)
        nn.init.constant_(self.hidden3.bias, 0)
        nn.init.constant_(self.hidden4.bias, 0)
        nn.init.constant_(self.hidden5.bias, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.fc(x)
        return x

class LargeMLPv10(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv10, self).__init__()
        self.hidden = nn.Linear(n_features[0], 8192)
        self.hidden2 = nn.Linear(8192, 8192)
        self.hidden3 = nn.Linear(8192, 8192)
        self.fc = nn.Linear(8192, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x
