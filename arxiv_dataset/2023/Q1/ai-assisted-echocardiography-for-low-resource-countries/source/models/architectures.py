import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ShrapML_replication_v00(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 1):
        """
        Boice, Emily N., Sofia I. Hernandez-Torres, and Eric J. Snider. 2022.
        "Comparison of Ultrasound Image Classifier Deep Learning Algorithms for Shrapnel Detection"
        Journal of Imaging 8, no. 5: 140. https://doi.org/10.3390/jimaging8050140

        emily.n.boice.ctr@mail.mil (E.N.B.);
        sofia.i.hernandeztorres.ctr@mail.mil (S.I.H.-T.)
        eric.j.snider3.civ@mail.mil
        """
        super(ShrapML_replication_v00, self).__init__()
        self.name = 'ShrapML_replication_v00'
        self.softmax = nn.Softmax(dim=1)  # along the column (for linear output)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=3, stride=1, padding=0),
            #             nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=0),
            #             nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 15, kernel_size=3, stride=1, padding=0),
            #             nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(15, 20, kernel_size=3, stride=1, padding=0),
            #             nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(20, 25, kernel_size=3, stride=1, padding=0),
            #             nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(25, 30, kernel_size=3, stride=1, padding=0),
            #             nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            #             nn.Dropout(0.5),
            # nn.Linear(9216, 4096),
            nn.Linear(1080, 214),
            # nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            #             nn.Dropout(0.5),
            nn.Linear(214, num_classes),
            # nn.ReLU()
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        #         print(f'layer6.shape(): {x.size()}')
        x = F.dropout(x, p=0.3641)  # dropout was included to combat overfitting
        x = self.flatten(x)  # flatten.shape(): torch.Size([25, 1080])
        x = self.fc(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class Block(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_planes, out_planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes // 16, out_planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out


class EfficientNet_source0(nn.Module):
    """
    https://github.com/TommyLitlle/AS-OCT-AFSNet/blob/main/efficientnet.py
    Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
    https://arxiv.org/abs/1905.11946?context=stat.ML
    https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    """

    def __init__(self, cfg, num_classes: int = 3, in_channels: int = 3):
        super(EfficientNet_source0, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1][1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.avg_pool2d(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class fire_(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire_, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet_source0(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        """
        SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
        Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
        [v4] Fri, 4 Nov 2016 21:26:08 UTC (533 KB)
        https://arxiv.org/abs/1602.07360
        https://github.com/gsp-27/pytorch_Squeezenet/blob/master/model.py
        https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
        https://github.com/Marcovaldong/LightModels/blob/master/models/shufflenet.py
        """
        super(SqueezeNet_source0, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1)  # 32
        # Input: (N, C_{in}, H_{in}, W_{in})o (C_{in}, H_{in}, W_{in})
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16
        self.fire2 = fire_(96, 16, 64)
        self.fire3 = fire_(128, 16, 64)
        self.fire4 = fire_(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8
        self.fire5 = fire_(256, 32, 128)
        self.fire6 = fire_(256, 48, 192)
        self.fire7 = fire_(384, 48, 192)
        self.fire8 = fire_(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4
        self.fire9 = fire_(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.LogSoftmax(dim=1)
        self.linear1 = nn.Linear(160, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=1)
        #         print(f'X0.shape(): {x.size()}')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        #         print(f'X?.shape(): {x.size()}')
        return x


# def fire_layer(inp, s, e):
#     f = fire(inp, s, e)
#     return f

# def squeezenet(pretrained=False):
#     net = SqueezeNet()
#     # inp = Variable(torch.randn(64,3,32,32))
#     # out = net.forward(inp)
#     # print(out.size())
#     return net


class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


# initial picture size = 256
class SqueezeNet_source1(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        """
        SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
        Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
        [v4] Fri, 4 Nov 2016 21:26:08 UTC (533 KB)
        https://github.com/thuBingo/Squeezenet_pytorch/blob/master/squeezenet.py
        """
        super(SqueezeNet_source1, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1)  # 128x128
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.fire9 = fire(512, 64, 256)
        self.softmax = nn.LogSoftmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x1 = self.fire2(x)
        x2 = self.fire3(x1)
        x = self.fire4(x1 + x2)
        x3 = self.maxpool2(x)
        x4 = self.fire5(x3)
        x5 = self.fire6(x3 + x4)
        x6 = self.fire7(x5)
        x = self.fire8(x5 + x6)
        x7 = self.maxpool3(x)
        x8 = self.fire9(x7)
        x = self.classifier(x7 + x8)
        x = self.softmax(x)
        return x.view(-1, self.num_classes)


# def squeezenet(pretrained=False):
#     net = SqueezeNet()
#     # inp = Variable(torch.randn(32,3,256,256))
#     # out = net.forward(inp)
#     # print(out.size())
#     return net


class Fire(nn.Module):
    """Fire Module based on SqueezeNet"""

    def __init__(
            self,
            in_channels,
            squeeze_channels,
            e1x1_channels,
            e3x3_channels
    ):
        super(Fire, self).__init__()

        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.e1x1_channels = e1x1_channels
        self.e3x3_channels = e3x3_channels

        self.squeeze_layer = self.get_squeeze_layer()
        self.expand_1x1_layer = self.get_expand_1x1_layer()
        self.expand_3x3_layer = self.get_expand_3x3_layer()

    def get_squeeze_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.in_channels,
                                self.squeeze_channels,
                                kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_1x1_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e1x1_channels,
                                kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_3x3_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e3x3_channels,
                                kernel_size=3,
                                padding=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.squeeze_layer(x)
        return torch.cat([
            self.expand_1x1_layer(y),
            self.expand_3x3_layer(y)
        ], 1)


class SqueezeNet_source2(nn.Module):
    """SqueezeNet1.1"""

    def __init__(
            self,
            channels,
            class_count
    ):
        super(SqueezeNet_source2, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.features = self.get_features()
        self.classifier = self.get_classifier()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)

                else:
                    init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_features(self):
        layers = []

        # in_channels = self.channels, out_channels = 64
        # kernel_size = 3x3, stride = 2
        layers.append(nn.Conv2d(self.channels, 64, kernel_size=3, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 64, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        layers.append(Fire(64, 16, 64, 64))

        # in_channels = 128, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        layers.append(Fire(128, 16, 64, 64))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 128, squeeze_channels = 32
        # e1x1_channels = 128, e3x3_channels = 128 -> out_channels = 256
        layers.append(Fire(128, 32, 128, 128))

        # in_channels = 256, squeeze_channels = 32
        # e1x1_channels = 128, e3x3_channels = 128 -> out_channels = 256
        layers.append(Fire(256, 32, 128, 128))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 256, squeeze_channels = 48
        # e1x1_channels = 192, e3x3_channels = 192 -> out_channels = 384
        layers.append(Fire(256, 48, 192, 192))

        # in_channels = 384, squeeze_channels = 48
        # e1x1_channels = 192, e3x3_channels = 192 -> out_channels = 384
        layers.append(Fire(384, 48, 192, 192))

        # in_channels = 384, squeeze_channels = 64
        # e1x1_channels = 256, e3x3_channels = 256 -> out_channels = 512
        layers.append(Fire(384, 64, 256, 256))

        # in_channels = 512, squeeze_channels = 64
        # e1x1_channels = 256, e3x3_channels = 256 -> out_channels = 512
        layers.append(Fire(512, 64, 256, 256))

        return nn.Sequential(*layers)

    def get_classifier(self):
        layers = []

        self.final_conv = nn.Conv2d(512, self.class_count, kernel_size=1)

        layers.append(nn.Dropout())
        layers.append(self.final_conv)
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.AvgPool2d(13, stride=1))

        # layers.append(nn.AdaptiveAvgPool2d((1, 1))) # Use Adaptive Average Pooling for random input image size
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.features(x)
        #         print(f'x.shape(): {x.size()}')
        x = self.classifier(x)
        x = x.view(x.size(0), self.class_count)
        return x


class LeNet5_source00(nn.Module):
    def __init__(self, n_classes=2):
        """
        https://medium.datadriveninvestor.com/cnn-architectures-from-scratch-c04d66ac20c2
        """
        super(LeNet5_source00, self).__init__()
        self.name = 'LeNet5_source00'
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(75000, 84)
        self.linear2 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.tanh(self.conv3(x))
        x = self.flatten(x)  # x = x.reshape(x.shape[0],-1) #75000
        x = self.tanh(self.linear1(x))
        x = self.linear2(x)
        return x


class LeNet5_source01(nn.Module):
    def __init__(self, n_classes=2):
        """
        https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
        """
        super(LeNet5_source01, self).__init__()
        self.name = 'LeNet5_source01'

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(13456, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class LeNet5_source02(nn.Module):
    def __init__(self, n_classes=2):
        """
        https://github.com/lychengr3x/LeNet-5-Implementation-Using-Pytorch/blob/master/LeNet-5%20Implementation%20Using%20Pytorch.ipynb
        """
        super(LeNet5_source02, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(16*5*5, 120)
        self.fc1 = nn.Linear(14400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        x = torch.squeeze(x, dim=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.flatten(x)  # x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_source00(nn.Module):
    def __init__(self, n_classes=2):
        """
        AlexNet
        https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        """
        super(AlexNet_source00, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            #             nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            # nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        # print(f'x.shape(): {x.size()}')
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f'x.shape(): {x.size()}')
        x = self.classifier(x)
        return x


class AlexNet_source01(nn.Module):
    def __init__(self, n_classes=2):
        """
        AlexNet
        https://blog.paperspace.com/alexnet-pytorch/
        """
        super(AlexNet_source01, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(9216, 4096),
            nn.Linear(1024, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, n_classes))

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        # print(f'x.shape(): {x.size()}')
        # x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AlexNet_source02(nn.Module):
    def __init__(self, n_classes=2):
        """
        AlexNet
        https://developpaper.com/example-of-pytorch-implementing-alexnet/
        """
        super(AlexNet_source02, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # nn.Linear(in_features=256*6*6,out_features=4096),
            nn.Linear(in_features=2304, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=n_classes),
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.feature_extraction(x)
        # print(f'x.shape(): {x.size()}')
        # x = x.view(x.size(0),256*6*6)
        x = torch.flatten(x, 1)
        # print(f'x.shape(): {x.size()}')
        x = self.classifier(x)
        return x


class AlexNet_source03(nn.Module):

    def __init__(self, num_classes=1000, in_channels=3):
        """
        Fetal Region Localisation using PyTorch and Soft Proposal Networks (paper: https://arxiv.org/abs/1808.00793)
        https://github.com/ntoussaint/fetalnav/blob/master/fetalnav/models/alexnet.py
        """
        super(AlexNet_source03, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            #             nn.Linear(256 * 6 * 6, 4096),
            nn.Linear(2304, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        #         x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }
# def alexnet(pretrained=False, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = AlexNet_source03(**kwargs)
#     if pretrained:
#         model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['alexnet']))
#     return model


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        """
        MobileNetV1
        References:
        https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py
        https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69
        """
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw: depthwise convolution
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw: pointwise convolution
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


################################
##### MobileNetV2 architecture
def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            # depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )


def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            # dw
            dwise_conv(hidden_dim, stride=stride),
            # pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, n_classes=1000):
        """
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/abs/1801.04381
        https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV2.py
        """
        super(MobileNetV2, self).__init__()

        self.configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x


################################
##### ShuffleNetV1 architecture
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N, g, C/g, H, W] -> [N, C/g, g, H, W] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(out_planes / 4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = functions.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = functions.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = functions.relu(torch.cat([out, res], 1)) if self.stride == 2 else functions.relu(out + res)
        return out


class ShuffleNetV1(nn.Module):
    def __init__(self, cfg):
        """
        ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
        Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun [v2] Thu, 7 Dec 2017 18:06:34 UTC (110 KB)
        https://arxiv.org/abs/1707.01083
        https://github.com/nrjanjanam/shufflenet-v1-pytorch/blob/main/code/ShuffleNet_Implementation.ipynb

        Usage:

        def ShuffleNetG2():
        cfg = {'out_planes': [200, 400, 800],
               'num_blocks': [4, 8, 4],
               'groups': 2
               }
        return ShuffleNet(cfg)

        def ShuffleNetV1_G3():
        cfg = {'out_planes': [240, 480, 960],
               'num_blocks': [4, 8, 4],
               'groups': 3
               }
        return ShuffleNetV1(cfg)

        Errors:
        > RuntimeError: Given groups=1, weight of size [24, 3, 1, 1],
        expected input[13, 1, 128, 128] to have 3 channels, but got 1 channels instead

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net3.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)
        """
        super(ShuffleNetV1, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']
        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], 2)  # 2 as there are 2 classes

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 4)
        x = out.view(x.size(0), -1)
        x = self.linear(x)
        return x


################################
##### ShuffleNetV2 architecture
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        """
        ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
        Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun [v1] Mon, 30 Jul 2018 04:18:25 UTC (4,805 KB)
        https://arxiv.org/abs/1807.11164
        https://github.com/ericsun99/Shufflenet-v2-Pytorch/blob/master/ShuffleNetV2.py

        > RuntimeError: Given groups=1, weight of size [24, 3, 3, 3], expected input[13, 1, 128, 128] to have 3 channels, but got 1 channels instead

        """
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model


class TrompNetV1(nn.Module):
    def __init__(self, input_pixel_size, n_batch_size_of_clips, n_frames_per_clip, n_classes=2):
        """
        Simple Video classifier by Tromp et al. 2022. DOI https://doi.org/10.1016/S2589-7500(21)00235-1
        The first classifier was a supervised CNN, composed of
            * four convolutional layers,
            * a dense layer, and
            * a softmax output layer.
        This model was trained with a categorical cross-entropy loss function.

        V1 was implemented by Miguel Xochicale based on
            Boice, Emily N., Sofia I. Hernandez-Torres, and Eric J. Snider. 2022.
            "Comparison of Ultrasound Image Classifier Deep Learning Algorithms for Shrapnel Detection"
            Journal of Imaging 8, no. 5: 140.
            DOI: https://doi.org/10.3390/jimaging8050140

        Args:
            input_pixel_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height) [e.g. 128, 128].
            n_batch_size_of_clips: (self explanatory)
            n_frames_per_clip: (self explanatory)
            n_classes: number of output classes
        """
        super(TrompNetV1, self).__init__()
        self.name = 'TrompNetV1'

        self.input_pixel_size = input_pixel_size  # [128, 128]
        self.n_batch_size_of_clips = n_batch_size_of_clips  # BATCH_SIZE_OF_CLIPS
        self.n_frames_per_clip = n_frames_per_clip  # NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP
        self.n_classes = n_classes
        self.n_features = np.prod(self.input_pixel_size) * self.n_frames_per_clip

        # self.n_batch_size_of_clips
        self.conv0 = nn.Conv2d(in_channels=1,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),  # H_{in}/strideA, W_{in}/strideB
                               padding=(0, 0),
                               # dilation=(0, 0)
                               )
        # Input: (N,Cin,Hin,Win)(N, C_{in}, H_{in}, W_{in})
        # Output: (N,Cout,Hout,Wout)(N, C_{out}, H_{out}, W_{out})
        # N is a batch size

        self.conv1 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),  # H_{in}/strideA, W_{in}/strideB
                               padding=(0, 0),
                               # dilation=(0, 0)
                               )

        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(1, 1),  # H_{in}/strideA, W_{in}/strideB
                               padding=(0, 0),
                               # dilation=(0, 0)
                               )

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=10,
                               kernel_size=(3, 3),
                               stride=(1, 1),  # H_{in}/strideA, W_{in}/strideB
                               padding=(0, 0),
                               # dilation=(0, 0)
                               )

        self.flatten = nn.Flatten()
        #         self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

        self.fc0 = nn.Linear(in_features=144000, out_features=self.n_classes)

        ### Softmax
        # self.softmax = nn.Softmax() # UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
        # self.softmax = nn.Softmax(dim=0) #along row
        self.softmax = nn.Softmax(dim=1)  # along the column (for linear output)
        # https://discuss.pytorch.org/t/implicit-dimension-choice-for-softmax-warning/12314/12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') # torch.Size([20, 1, 1, 128, 128])
        x = torch.squeeze(x, dim=1)
        # x = torch.unsqueeze(x,dim=1)
        # print(f'x.shape(): {x.size()}') # torch.Size([20, 1, 128, 128])

        # print(f'X.shape(): {x.size()}')
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(f'conv3(x): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape(): torch.Size([4, 983040])
        x = self.fc0(x)
        x = F.dropout(x, p=0.5)  # dropout was included to combat overfitting
        # print(f'fc0(): {x.size()}')
        #         x = self.relu(x)
        x = self.softmax(x)
        # print(f'x.shape(): {x.size()}')
        return x


class basicVGG3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, n_frames_per_clip: int = 1):
        """
        Simple Video classifier to classify into two classes:
        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height) [e.g. 128, 128].
            n_classes: number of output classes

        References:
        * downsample_factors: https://github.com/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/vgg3d.py
        """

        super(basicVGG3D, self).__init__()
        self.name = 'basicVGG3D'

        self.in_channels = in_channels
        self.num_classes = num_classes
        #         self.n_frames_per_clip = n_frames_per_clip
        #         self.n_features = np.prod(self.input_size) * self.n_frames_per_clip

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #         self.softmax = nn.Softmax()
        self.softmax = nn.LogSoftmax(dim=1)

        # downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        downsample_factors = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
        # stride=(depth/val0, height/val1, width/val2)

        # NOTES
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        # IN: [N,Cin,D,H,W]; OUT: (N,Cout,Dout,Hout,Wout)
        # [batch_size, channels, depth, height, width].

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=16,
                      kernel_size=(1, 1, 1),  ## (-depth, -height, -width)
                      stride=downsample_factors[0],  ##(depth/val0, height/val1, width/val2)
                      padding=(0, 0, 0),
                      bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32,
                      kernel_size=(1, 1, 1),  ## (-depth, -height, -width)
                      stride=downsample_factors[1],  ##(depth/val0, height/val1, width/val2)
                      padding=(0, 0, 0),
                      bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64,
                      kernel_size=(1, 1, 1),  ## (-depth, -height, -width)
                      stride=downsample_factors[2],  ##(depth/val0, height/val1, width/val2)
                      padding=(0, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128,
                      kernel_size=(1, 1, 1),  ## (-depth, -height, -width)
                      stride=downsample_factors[3],  ##(depth/val0, height/val1, width/val2)
                      padding=(0, 0, 0),
                      bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        )

        # #         self.fc0 = nn.Linear(in_features=2097152, out_features=500)  # 128x128
        #         self.fc0 = nn.Linear(in_features=1048576, out_features=500) #
        #         # self.fc0 = nn.Linear(in_features=4194304, out_features=500) #128x128
        #         self.fc2 = nn.Linear(in_features=500, out_features=self.num_classes)
        #         # self.fc1 = nn.Linear(in_features=2048, out_features=self.num_classes)

        # 1048576
        # 16384
        # 524288
        # 8192
        # 2097152
        self.fc0 = nn.Linear(in_features=2097152, out_features=self.num_classes)  #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([25, 1, 1, 128, 128])

        x = self.conv0(x)
        # print(f'x.shape(): {x.size()}') #x.shape(): x.shape(): torch.Size([2, 64, 60, 128, 128]) with kernel_size=(1, 1, 1)
        # print(f'x.shape(): {x.size()}') #x.shape():torch.Size([2, 64, 51, 29, 29]) with kernel_size=(10, 100, 100)
        # print(f'conv0.size(): {x.size()}')

        x = self.conv1(x)
        # print(f'x.shape(): {x.size()}') with kernel_size=(1, 10, 10) #x.shape(): torch.Size([2, 32, 60, 20, 20])
        # print(f'conv1.size(): {x.size()}')

        x = self.conv2(x)
        # print(f'x.shape(): {x.size()}') with kernel_size=(1, 10, 10) #x.shape(): torch.Size([2, 32, 60, 20, 20])
        # print(f'conv2.size(): {x.size()}')

        x = self.conv3(x)
        ## print(f'x.shape(): {x.size()}') with kernel_size=(1, 10, 10) #x.shape(): torch.Size([2, 32, 60, 20, 20])
        # print(f'conv3.size(): {x.size()}')

        # x = self.pool0(x)
        # print(f'x.pool0..shape(): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape():
        x = self.fc0(x)
        # print(f'self.fc0(x): {x.size()}') #x.shape():

        x = F.relu(x)
        x = F.dropout(x, p=0.5)  # dropout was included to combat overfitting
        # x = self.softmax(x)
        # x = self.sigmoid(x)

        # x = self.fc2(x)
        # print(f'x.shape(): {x.size()}')  #x.shape(): torch.Size([4, 2])

        return x


################################
##### Define basicVGG architecture
class basicVGG(nn.Module):

    def __init__(self, input_size, n_frames_per_clip, n_classes=2):
        """
        Simple Video classifier to classify into two classes:
        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height) [e.g. 128, 128].
            n_classes: number of output classes
        """

        super(basicVGG, self).__init__()
        self.name = 'basicVGG'
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_frames_per_clip = n_frames_per_clip
        self.n_features = np.prod(self.input_size) * self.n_frames_per_clip

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_classes),
            # nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth,channels, height, width]
        # print(f'x.shape(): {x.size()}')

        x = self.classifier(x)

        return x


class basicVGG2D_13layers(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, n_frames_per_clip: int = 1):
        """

        References:
        * https://blog.paperspace.com/vgg-from-scratch-pytorch/
        """

        super(basicVGG2D_13layers, self).__init__()
        self.name = 'basicVGG2D_13layers'

        self.in_channels = in_channels
        self.num_classes = num_classes
        #         self.n_frames_per_clip = n_frames_per_clip
        #         self.n_features = np.prod(self.input_size) * self.n_frames_per_clip
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(262144, 4096), #layer1,2 #RuntimeError: CUDA out of memory.
            nn.Linear(8192, 4096),  # 13layers
            ##nn.Linear(7*7*512, 4096), #DEFAULT
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([25, 1, 1, 128, 128])
        x = torch.squeeze(x, dim=1)
        # print(f'x.shape(): {x.size()}') #        x.shape(): torch.Size([5, 1, 128, 128])

        x = self.layer1(x)
        # print(f'layer1.size(): {x.size()}')
        x = self.layer2(x)
        # print(f'layer2.size(): {x.size()}')
        x = self.layer3(x)
        # print(f'layer3.size(): {x.size()}')
        x = self.layer4(x)
        # print(f'layer4.size(): {x.size()}')
        x = self.layer5(x)
        # print(f'layer5.size(): {x.size()}')
        x = self.layer6(x)
        # print(f'layer6.size(): {x.size()}')
        x = self.layer7(x)
        # print(f'layer7.size(): {x.size()}')
        x = self.layer8(x)
        # print(f'layer8.size(): {x.size()}')
        x = self.layer9(x)
        # print(f'layer9.size(): {x.size()}')
        x = self.layer10(x)
        # print(f'layer10.size(): {x.size()}')
        x = self.layer11(x)
        # print(f'layer11.size(): {x.size()}')
        x = self.layer12(x)
        # print(f'layer12.size(): {x.size()}')
        x = self.layer13(x)
        # print(f'layer13.size(): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape():
        x = self.fc0(x)
        # print(f'self.fc0(x): {x.size()}') #x.shape():
        x = self.fc1(x)
        # print(f'self.fc1(x): {x.size()}') #x.shape():
        x = self.fc2(x)
        # print(f'self.fc2(x): {x.size()}') #x.shape():
        # print(f'x.shape(): {x.size()}')  #x.shape(): torch.Size([4, 2])

        return x


class basicVGG2D_07layers(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, n_frames_per_clip: int = 1):
        """

        References:
        * https://blog.paperspace.com/vgg-from-scratch-pytorch/
        """

        super(basicVGG2D_07layers, self).__init__()
        self.name = 'basicVGG2D_07layers'

        self.in_channels = in_channels
        self.num_classes = num_classes
        #         self.n_frames_per_clip = n_frames_per_clip
        #         self.n_features = np.prod(self.input_size) * self.n_frames_per_clip
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(262144, 4096), #layer1,2 #RuntimeError: CUDA out of memory.
            # nn.Linear(131072, 4096), #layer1,2,3,4
            nn.Linear(65536, 4096),  # layer1,2,3,4,5,6,7
            # nn.Linear(8192, 4096), #ALL nn.Linear(8192, 4096) #nn.Linear(7*7*512, 4096)
            ##nn.Linear(7*7*512, 4096), #DEFAULT #
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([25, 1, 1, 128, 128])
        x = torch.squeeze(x, dim=1)
        # print(f'x.shape(): {x.size()}') #        x.shape(): torch.Size([5, 1, 128, 128])

        x = self.layer1(x)
        # print(f'layer1.size(): {x.size()}')
        x = self.layer2(x)
        # print(f'layer2.size(): {x.size()}')
        x = self.layer3(x)
        # print(f'layer3.size(): {x.size()}')
        x = self.layer4(x)
        # print(f'layer4.size(): {x.size()}')
        x = self.layer5(x)
        # print(f'layer5.size(): {x.size()}')
        x = self.layer6(x)
        # print(f'layer6.size(): {x.size()}')
        x = self.layer7(x)
        # print(f'layer7.size(): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape():
        x = self.fc0(x)
        # print(f'self.fc0(x): {x.size()}') #x.shape():
        x = self.fc1(x)
        # print(f'self.fc1(x): {x.size()}') #x.shape():
        x = self.fc2(x)
        # print(f'self.fc2(x): {x.size()}') #x.shape():
        # print(f'x.shape(): {x.size()}')  #x.shape(): torch.Size([4, 2])

        return x


class basicVGG2D_04layers(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, n_frames_per_clip: int = 1):
        """

        References:
        * https://blog.paperspace.com/vgg-from-scratch-pytorch/
        """

        super(basicVGG2D_04layers, self).__init__()
        self.name = 'basicVGG2D_04layers'

        self.in_channels = in_channels
        self.num_classes = num_classes
        #         self.n_frames_per_clip = n_frames_per_clip
        #         self.n_features = np.prod(self.input_size) * self.n_frames_per_clip
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(131072, 4096),  # layer1,2,3,4
            ##nn.Linear(7*7*512, 4096), #DEFAULT #
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([25, 1, 1, 128, 128])
        x = torch.squeeze(x, dim=1)
        # print(f'x.shape(): {x.size()}') #        x.shape(): torch.Size([5, 1, 128, 128])

        x = self.layer1(x)
        # print(f'layer1.size(): {x.size()}')
        x = self.layer2(x)
        # print(f'layer2.size(): {x.size()}')
        x = self.layer3(x)
        # print(f'layer3.size(): {x.size()}')
        x = self.layer4(x)
        # print(f'layer4.size(): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape():
        x = self.fc0(x)
        # print(f'self.fc0(x): {x.size()}') #x.shape():
        x = self.fc1(x)
        # print(f'self.fc1(x): {x.size()}') #x.shape():
        x = self.fc2(x)
        # print(f'self.fc2(x): {x.size()}') #x.shape():
        # print(f'x.shape(): {x.size()}')  #x.shape(): torch.Size([4, 2])

        return x


class basicVGG2D_NNlayers_sequence(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, n_frames_per_clip: int = 1):
        """

        References:
        * https://blog.paperspace.com/vgg-from-scratch-pytorch/
        """

        super(basicVGG2D_NNlayers_sequence, self).__init__()
        self.name = 'basicVGG2D_NNlayers_sequence'

        self.in_channels = in_channels
        self.num_classes = num_classes
        #         self.n_frames_per_clip = n_frames_per_clip
        #         self.n_features = np.prod(self.input_size) * self.n_frames_per_clip
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #         self.layer5 = nn.Sequential(
        #             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU())
        #         self.layer6 = nn.Sequential(
        #             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU())
        #         self.layer7 = nn.Sequential(
        #             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU(),
        #             nn.MaxPool2d(kernel_size=2, stride=2))
        #         self.layer8 = nn.Sequential(
        #             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.layer9 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.layer10 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(),
        #             nn.MaxPool2d(kernel_size=2, stride=2))
        #         self.layer11 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.layer12 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU())
        #         self.layer13 = nn.Sequential(
        #             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(),
        #             nn.MaxPool2d(kernel_size=2, stride=2))

        #         self.fc0 = nn.Sequential(
        #             nn.Dropout(0.5),
        #             nn.Linear(16384, self.num_classes),
        #             nn.ReLU()) #INDIVICUALfc0 DONTTRAIN FOR THE 4 and 7 LAYER CASE

        #         self.fc0 = nn.Sequential(
        #             nn.Dropout(0.5),
        #             nn.Linear(16384, 4096),  # layer1,2,3,4
        #             nn.ReLU())
        #         self.fc1 = nn.Sequential(
        #             nn.Dropout(0.5),
        #             nn.Linear(4096, self.num_classes),
        #             nn.ReLU()) ##INDIVICUALfc0andfc1 DONTR TRIN FLAT

        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear( 16*128, 4096),  # 13layers
            # nn.Linear(16384, 4096),  # layer1,2,3,4,5,6,7
            nn.Linear(256 * 128, 4096),  # layer1to4
            ##nn.Linear(7*7*512, 4096), #DEFAULT #
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth, channels, height, width]
        # print(f'x.shape(): {x.size()}') #x.shape(): torch.Size([25, 1, 1, 128, 128])
        x = torch.squeeze(x, dim=1)
        # print(f'x.shape(): {x.size()}') #        x.shape(): torch.Size([5, 1, 128, 128])

        x = self.layer1(x)
        # print(f'layer1.size(): {x.size()}')
        x = self.layer2(x)
        # print(f'layer2.size(): {x.size()}')
        x = self.layer3(x)
        # print(f'layer3.size(): {x.size()}')
        x = self.layer4(x)
        #         #print(f'layer4.size(): {x.size()}')
        #         x = self.layer5(x)
        #         # print(f'layer5.size(): {x.size()}')
        #         x = self.layer6(x)
        #         # print(f'layer6.size(): {x.size()}')
        #         x = self.layer7(x)
        #         # print(f'layer7.size(): {x.size()}')
        #         x = self.layer8(x)
        #         # print(f'layer8.size(): {x.size()}')
        #         x = self.layer9(x)
        #         # print(f'layer9.size(): {x.size()}')
        #         x = self.layer10(x)
        #         # print(f'layer10.size(): {x.size()}')
        #         x = self.layer11(x)
        #         # print(f'layer11.size(): {x.size()}')
        #         x = self.layer12(x)
        #         # print(f'layer12.size(): {x.size()}')
        #         x = self.layer13(x)
        #         # print(f'layer13.size(): {x.size()}')

        x = self.flatten(x)
        # print(f'self.flatten(x) size() {x.size()}')  # x.shape():
        x = self.fc0(x)
        # print(f'self.fc0(x): {x.size()}') #x.shape():
        x = self.fc1(x)
        # print(f'self.fc1(x): {x.size()}') #x.shape():
        x = self.fc2(x)
        # print(f'self.fc2(x): {x.size()}') #x.shape():
        # print(f'x.shape(): {x.size()}')  #x.shape(): torch.Size([4, 2])

        return x


class basicVGGNet(nn.Module):

    def __init__(self, tensor_shape_size, n_classes=2, cnn_channels=(1, 16, 32)):
        """
        Simple Visual Geometry Group Network (VGGNet) to classify two US image classes (background and 4CV).

        Args:
            tensor_shape_size: [Batch_clips, Depth, Channels, Height, Depth]

        """
        super(basicVGGNet, self).__init__()
        self.name = 'basicVGGNet'

        self.tensor_shape_size = tensor_shape_size
        self.n_classes = n_classes

        # define the CNN
        self.n_output_channels = cnn_channels  ##  self.n_output_channels::: (1, 16, 32)
        self.kernel_size = (3,) * (len(cnn_channels) - 1)  ## self.kernel_size::: (3, 3)

        self.n_batch_size_of_clip_numbers = self.tensor_shape_size[0]
        self.n_frames_per_clip = self.tensor_shape_size[1]
        self.n_number_of_image_channels = self.tensor_shape_size[2]
        self.input_shape_tensor = self.n_batch_size_of_clip_numbers * self.n_frames_per_clip * self.n_number_of_image_channels

        self.conv1 = nn.Conv3d(in_channels=self.n_number_of_image_channels, out_channels=64,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
                               )
        # IN: [N,Cin,D,H,W]; OUT: (N,Cout,Dout,Hout,Wout)
        # [batch_size, channels, depth, height, width].

        self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.fc1 = nn.Linear(in_features=62914560, out_features=self.n_classes)

    def forward(self, x):
        print(f'x.shape(): {x.size()}')  # x.shape(): torch.Size([10, 60, 1, 128, 128])
        x = torch.permute(x, (0, 2, 1, 3, 4))  ##[batch_size, channels, depth, height, width]
        print(f'x.shape(): {x.size()}')  # x.shape(): torch.Size([10, 1, 60, 128, 128])
        # x = F.relu(self.conv1(x))
        # x = self.maxpool3d(x)
        # x = x.reshape(x.shape[0], -1)
        # x = F.dropout(x, p=0.5) #dropout was included to combat overfitting
        # x = self.fc1(x)

        return x


class BasicCNNClassifier(nn.Module):

    def __init__(self, input_size, n_classes=2):
        """
        Simple Video classifier to classify into two classes:

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height) [e.g. 128, 128].
            n_classes: number of output classes
        """

        super(BasicCNNClassifier, self).__init__()
        self.name = 'BasicCNNClassifier'
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_frames_per_clip = 60
        self.n_features = np.prod(self.input_size) * self.n_frames_per_clip

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        # print(f'  x.size():::::::  {x.size()}')  #   x.size():::::::  torch.Size([2, 2])
        # print(x)
        # tensor([[0.1271, 0.6632],
        #        [0.3063, 0.5489]], device='cuda:0', grad_fn= < SigmoidBackward >)
        return x


class SimpleVideoClassifier(nn.Module):

    def __init__(self, input_size, n_output_classes=2):
        """
        Simple Video classifier to classify into two classes:

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super().__init__()

        self.name = 'SimpleVideoClassifier'
        self.input_size = input_size
        self.n_output_classes = n_output_classes

        n_features = np.prod(self.input_size)
        n_frames = self.input_size[1]
        n_spatial_features = np.prod(self.input_size[-2:])
        n_spatial_features_out = 32

        # extract features from each frame
        self.frame_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_spatial_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_spatial_features_out),
            nn.ReLU()
        )

        n_temporal_features = n_spatial_features_out * n_frames

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_temporal_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.n_output_classes),
            # nn.Sigmoid(),
        )

    def get_name(self):
        return self.name

    def forward(self, data):
        n_frames = data.shape[2]
        features = []
        for f in range(n_frames):
            feat_i = self.frame_features(data[:, :, f, ...])
            features.append(feat_i)

        feature_vector = torch.stack(features, dim=1)

        out = self.classifier(feature_vector)
        return out
