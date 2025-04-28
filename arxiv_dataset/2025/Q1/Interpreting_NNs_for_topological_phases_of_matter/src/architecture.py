import torch.nn as nn
from torch import flatten


class CNN_SSH(nn.Module):
    def __init__(self):
        super(CNN_SSH, self).__init__()

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(5, 7, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(7, 9, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(9, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class CNNv2_SSH(nn.Module):
    def __init__(self):
        super(CNNv2_SSH, self).__init__()

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5), 
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(16, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class FC_SSH(nn.Module):
    def __init__(self):
        super(FC_SSH, self).__init__()

        self.linear_block_1 = nn.Sequential(
            nn.Linear(2500, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.linear_block_2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.linear_final = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )



    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_block_1(x)
        x = self.linear_block_2(x)
        x = self.linear_final(x)
        return x

class Linear_SSH(nn.Module):
    def __init__(self):
        super(Linear_SSH, self).__init__()

        self.linear_final = nn.Sequential(
            nn.Linear(2500, 2),
            nn.Softmax(dim=1)
        )



    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_final(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet_SSH(nn.Module):
    def __init__(self):
        super(ResNet_SSH, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # [BS, 64, 24, 24]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  # [BS, 64, 12, 12]
        
        self.layer1 = self._make_layer(64, 64, 2)  # [BS, 64, 12, 12]
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # [BS, 128, 6, 6]
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # [BS, 256, 3, 3]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [BS, 256, 1, 1]
        self.fc = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  # [BS, 64, 24, 24]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [BS, 64, 12, 12]
        
        x = self.layer1(x)  # [BS, 64, 12, 12]
        x = self.layer2(x)  # [BS, 128, 6, 6]
        x = self.layer3(x)  # [BS, 256, 3, 3]
        
        x = self.avgpool(x)  # [BS, 256, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x



# =============================================================================
#
# =============================================================================


class CNN_SSH_ThermEncod(nn.Module):
    def __init__(self, levels, in_layer_1=1, out_layer_1=7, out_layer_2=9):
        super(CNN_SSH_ThermEncod, self).__init__()
        self.in_layer_1 = in_layer_1
        self.out_layer_1 = out_layer_1
        self.out_layer_2 = out_layer_2

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_layer_1 * levels,
                out_channels=5 * levels,
                kernel_size=4,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(5 * levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                5 * levels, 7 * levels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(7 * levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(
                7 * levels, 9 * levels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            # W_out = ___
            nn.BatchNorm2d(9 * levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(9 * levels, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
