from CODE.Utils.package import *
#from Utils.augmentation import Augmentation
import torch.nn.init as init
from torch import nn, optim
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

############################################# SOTA1
class LSTMFCN(nn.Module):
    def __init__(self, input_shape=1, nb_classes=None, num_cells=64, **kwargs):
        super(LSTMFCN, self).__init__()
        seq_len = input_shape
        
        self.lstm = nn.LSTM(input_size=seq_len, hidden_size=num_cells, batch_first=True)
        self.dropout = nn.Dropout(0.8)
         # Convolutional block
        self.conv1 = nn.Conv1d(1, 128, kernel_size=8, padding="same", bias=True)#128
        init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(128)#128
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=True)#128 256
        init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(256)#256
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=True)#256
        init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
         # Final dense layer
        self.fc = nn.Linear(num_cells + 128, nb_classes)
    def forward(self, x):
        # LSTM part
        x1 = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x1)
        x_lstm = self.dropout(x_lstm[:, -1, :]) # We only use the output of the last LSTM cell
        # Convolutional part
        x_conv = F.relu(self.bn1(self.conv1(x)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = torch.flatten(self.global_avg_pool(x_conv), 1) 
        x = torch.cat((x_lstm, x_conv), dim=1)
        x = self.fc(x)
        return F.softmax(x, dim=1)
    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1).mean()
        return entropy

class LSTMFCN_Logits(nn.Module):
    def __init__(self, input_shape=1, nb_classes=None, num_cells=64, **kwargs):
        super(LSTMFCN_Logits, self).__init__()
        seq_len = input_shape
        
        self.lstm = nn.LSTM(input_size=seq_len, hidden_size=num_cells, batch_first=True)
        self.dropout = nn.Dropout(0.8)
         # Convolutional block
        self.conv1 = nn.Conv1d(1, 128, kernel_size=8, padding="same", bias=True)#128
        init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(128)#128
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=True)#128 256
        init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(256)#256
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=True)#256
        init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
         # Final dense layer
        self.fc = nn.Linear(num_cells + 128, nb_classes)
    def forward(self, x):
        # LSTM part
        x1 = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x1)
        x_lstm = self.dropout(x_lstm[:, -1, :]) # We only use the output of the last LSTM cell
        # Convolutional part
        x_conv = F.relu(self.bn1(self.conv1(x)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = torch.flatten(self.global_avg_pool(x_conv), 1) 
        x = torch.cat((x_lstm, x_conv), dim=1)
        x = self.fc(x)
        return x
    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1).mean()
        return entropy

##################################################resnet-18
class ResRoad(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResRoad, self).__init__()
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.downsample(x)
class MainRoad(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
    ):
        super(MainRoad, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResNetBlock, self).__init__()
        self.res = ResRoad(in_channels, out_channels, stride)

        self.layers = nn.ModuleList()
        self.layers.append(
            MainRoad(in_channels, out_channels, kernel_size, stride, padding)
        )
        self.layers.append(
            MainRoad(out_channels, out_channels, kernel_size, 1, padding)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.res(x)
        for layer in self.layers:
            x = layer(x)
        x += identity
        x = self.relu(x)
        return x
class ResNetBlock_deep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResNetBlock_deep, self).__init__()
        self.res = ResRoad(in_channels, out_channels)

        self.layers = nn.ModuleList()
        self.layers.append(MainRoad(in_channels, out_channels, 1, 1, 0))
        self.layers.append(
            MainRoad(out_channels, out_channels, kernel_size, stride, padding)
        )
        self.layers.append(MainRoad(out_channels, out_channels, 1, 1, 0))

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.res(x)
        for layer in self.layers:
            x = layer(x)
        x += identity
        x = self.relu(x)
        return x
class ClassifierResNet(nn.Module):
    def __init__(
        self,
        input_shape,
        nb_classes,
        channels_list,
        kernel_size_list,
        stride_list,
        deep=False,
    ):
        super(ClassifierResNet, self).__init__()
        input_channels = 1

        self.start = MainRoad(
            input_channels,
            channels_list[0],
            kernel_size=kernel_size_list[0],
            stride=stride_list[0],
            padding=(kernel_size_list[0] - 1) // 2,
        )

        self.resblock = ResNetBlock_deep if deep else ResNetBlock
        self.layers = nn.ModuleList()

        for idx in range(len(channels_list) - 1):
            in_channels = channels_list[idx]
            out_channels = channels_list[idx + 1]
            self.layers.append(
                self.resblock(
                    in_channels,
                    out_channels,
                    kernel_size_list[idx],
                    stride=stride_list[idx+1],
                    padding=(kernel_size_list[idx] - 1) // 2,
                )
            )
            in_channels = out_channels  

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels_list[-1], nb_classes)

    def forward(self, x):
        x = self.start(x)

        for layer in self.layers:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x
class ClassifierResNet18(ClassifierResNet):
    def __init__(self, input_shape, nb_classes, **kwargs):

        channels_list = [64, 64, 128, 128, 256, 256, 512, 512]
        kernel_size_list = [7, *[3] * 7]  
        stride_list = [2, 1, 1, 2, 1, 2, 1, 2, 1]

        super(ClassifierResNet18, self).__init__(
            input_shape=input_shape,
            nb_classes=nb_classes,
            channels_list=channels_list,
            kernel_size_list=kernel_size_list,
            stride_list=stride_list,  
            deep=False,  
        )
    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1).mean()
        return entropy

###############################################################MACNN
class Classifier_MACNN(nn.Module):
    def __init__(self, input_shape, nb_classes, *wargs, **kwargs):
        super(Classifier_MACNN, self).__init__()

        # Define your network layers here
        self.stack1 = self._make_stack(input_shape, 64, 2)
        self.pool1 = nn.MaxPool1d(3, 2, padding=1)
        self.stack2 = self._make_stack(192, 128, 2)
        self.pool2 = nn.MaxPool1d(3, 2, padding=1)
        self.stack3 = self._make_stack(384, 256, 2)
        # Add more stacks and pooling layers as required

        self.fc = nn.Linear(
            768, nb_classes
        )  # Adjust the input features according to your network

    def forward(self, x):
        x = self.stack1(x)
        x = self.pool1(x)
        x = self.stack2(x)
        x = self.pool2(x)
        x = self.stack3(x)
        # Add more stacks and pooling layers as required

        x = torch.mean(x, 2)  # Global Average Pooling
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

    def _make_stack(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(MACNNBlock(in_channels, out_channels))
            in_channels = out_channels * 3  
        return nn.Sequential(*layers)
    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1).mean()
        return entropy


class MACNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduce=16):
        super(MACNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=6,
            padding="same",
        )
        self.conv3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=12,
            padding="same",
        )
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.attention_fc1 = nn.Linear(out_channels * 3, int(out_channels * 3 / reduce))
        self.attention_fc2 = nn.Linear(int(out_channels * 3 / reduce), out_channels * 3)

    def forward(self, x):
        cov1 = self.conv1(x)
        cov2 = self.conv2(x)
        cov3 = self.conv3(x)

        x = torch.cat([cov1, cov2, cov3], 1)
        x = self.bn(x)
        x = F.relu(x)

        y = torch.mean(x, 2)
        y = F.relu(self.attention_fc1(y))
        y = torch.sigmoid(self.attention_fc2(y))
        y = y.view(
            y.shape[0], y.shape[1], -1
        )  # reshape to [batch_size, out_channels * 3, 1]

        return x * y
