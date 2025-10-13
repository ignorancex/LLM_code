import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Edited from From https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, region=False, n_regions=16):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if region:
            self.region_encoder = nn.Embedding(n_regions, 1024)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, coords=None):

        if coords is not None:
            region_encoding = self.region_encoder(coords)
            region_encoding = region_encoding.reshape((region_encoding.shape[0], 1, 32, 32))
            
            x = torch.cat((x, region_encoding), dim=1)


        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        """
        if coords is not None:
            #region_encoding = self.region_encoder((coords - 16) / (112- 16))
            region_encoding = self.region_encoder(coords).squeeze(1)
            out = torch.cat((out, region_encoding), dim=1)
        """
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, region=True, n_regions=16):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, region=region, n_regions=n_regions)

class AttributeClassifier(nn.Module):

    def __init__(self, num_classes: int = 1):
        super(AttributeClassifier, self).__init__()
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5,
                      padding="same"),  # in = 224*184
            nn.Dropout(0.1),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=6, out_channels=12,
                      kernel_size=5, padding="same"),
            nn.Dropout(0.1),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=12, out_channels=24,
                      kernel_size=5, padding="same"),
            nn.Dropout(0.1),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),
            # reduce more

            nn.Flatten(),  # 32 * 56 * 46
            nn.Linear(6144, 1000),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 100),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(100, self.num_classes),
        )

    def forward(self, x):
        return self.cnn(x)
