import torch
import torch.nn as nn


class AlexnetRegLog(nn.Module):            # 224
    """
    Create logistic regression on top of frozen features
    """
    def __init__(self, conv, num_labels):
        super(AlexnetRegLog, self).__init__()
        self.conv = conv
        s = 1
        if conv == 1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv == 2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv == 3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv == 4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv == 5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)

def forward_alexnet(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.module.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x


class ResnetCifarRegLog(nn.Module):   # 32

    def __init__(self, conv, num_labels):
        super(ResnetCifarRegLog, self).__init__()
        self.conv = conv
        s = 1
        if conv==1:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 4096
        elif conv == 2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 4096
        elif conv == 3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 4608
        elif conv == 4:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 4096
        elif conv == 5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=1)
            s = 4608
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


class ResnetRegLog(nn.Module):     # 224

    def __init__(self, conv, num_labels):
        super(ResnetRegLog, self).__init__()
        self.conv = conv
        s = 1
        if conv == 1:
            self.av_pool = nn.AvgPool2d(10, stride=10, padding=0)
            s = 7744
        elif conv == 2:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 6400
        elif conv == 3:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 6272
        elif conv == 4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 6400
        elif conv == 5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 4608
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


def forward_resnet18(x, model, conv):
    layer = {1: 'bn1', 2: 'layer1', 3: 'layer2', 4: 'layer3', 5: 'layer4'}
    for m in model.module._modules.items():
        # print(m) #('conv1', Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        # print(m[0]) #conv1
        x = m[1](x)
        if m[0] == layer[conv]:
            break
        else:
            continue
    return x
