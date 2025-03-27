import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import math
import torch
from torch.distributions import Bernoulli
import torch.nn.functional as F
import functools
from model.nets_factory import *


class VGG_shallow(nn.Module):
    def __init__(self, arch: object, num_classes=1000) -> object:
        super(VGG_shallow, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        # self.conv3_512a = self.__make_layer(512, arch[3])
        # self.conv3_512b = self.__make_layer(512, arch[4])

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
       
        # return F.softmax(self.fc3(out))
        return out


class VGG_deep_head(nn.Module):
    def __init__(self, arch: object, num_classes=1000) -> object:
        super(VGG_deep_head, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_512a(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
       
        # return F.softmax(self.fc3(out))
        return out


class VGG_deep_shape(nn.Module):
    def __init__(self, num_classes):
        super(VGG_deep_shape, self).__init__()
        self.padding = nn.ReflectionPad2d(3)
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0)
        self.norm0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        tmp_list = []
        for i in range(9):
            tmp_list += [ResnetBlock(dim=256, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_bias=False, use_dropout=False)]
        self.res_block_sequential = nn.Sequential(*tmp_list)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.res_block_sequential(x)
        ret = self.avgpool(x)
        return x, ret


class VGG_deep(nn.Module):
    def __init__(self, arch: object, num_classes=1000) -> object:
        super(VGG_deep, self).__init__()
        self.in_channels = 256
        # self.conv3_64 = self.__make_layer(64, arch[0])
        # self.conv3_128 = self.__make_layer(128, arch[1])
        # self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_512a(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
       
        # return F.softmax(self.fc3(out))
        return out


class VGG_Reconstructor_head(nn.Module):
    def __init__(self):
        super(VGG_Reconstructor_head, self).__init__()
        # self.dim_reduct_conv = nn.Conv2d(5888, 512, kernel_size=1)
        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.convt4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        self.convt5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm5 = nn.BatchNorm2d(64)
        self.padding = nn.ReflectionPad2d(3)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=7, padding=0)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.convt2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.convt3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.convt4(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.convt5(x)
        x = self.norm5(x)
        x = self.relu(x)

        x = self.padding(x)

        x = self.conv6(x)
        x = self.tanh(x)

        return x


class VGG_Reconstructor(nn.Module):
    def __init__(self):
        super(VGG_Reconstructor, self).__init__()
        # self.dim_reduct_conv = nn.Conv2d(5888, 512, kernel_size=1)
        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.convt4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        self.convt5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm5 = nn.BatchNorm2d(64)
        self.padding = nn.ReflectionPad2d(3)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=7, padding=0)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.convt2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.convt3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.convt4(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.convt5(x)
        x = self.norm5(x)
        x = self.relu(x)

        x = self.padding(x)

        x = self.conv6(x)
        x = self.tanh(x)

        return x


class VGG_Reconstructor_shape(nn.Module):
    def __init__(self):
        super(VGG_Reconstructor_shape, self).__init__()
        # self.dim_reduct_conv = nn.Conv2d(5888, 512, kernel_size=1)
        # self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1)
        # self.convt1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.norm1 = nn.BatchNorm2d(1024)
        # self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)
        # self.convt2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.norm2 = nn.BatchNorm2d(512)
        # self.conv3 = nn.Conv2d(512, 512, kernel_size=1)
        self.convt3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.convt4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.convt5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm5 = nn.BatchNorm2d(64)
        self.convt_tmp = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm_tmp = nn.BatchNorm2d(64)
        self.padding = nn.ReflectionPad2d(3)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=7, padding=0)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.convt4(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.convt5(x)
        x = self.norm5(x)
        x = self.relu(x)

        x = self.convt_tmp(x)
        x = self.norm_tmp(x)
        x = self.relu(x)

        x = self.padding(x)

        x = self.conv6(x)
        x = self.tanh(x)

        return x


class VGG_Discriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(VGG_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class VGG_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(VGG_Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1792, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def vgg_shallow(num_classes=1000):
    return VGG_shallow([1, 1, 2, 2, 2], num_classes=num_classes)


def vgg_deep_head(num_classes=1000):
    return  VGG_deep_head([1, 1, 2, 2, 2], num_classes=num_classes)


def vgg_deep_shape(num_classes=1000):
    return VGG_deep_shape(num_classes=num_classes)


def vgg_deep(num_classes=1000):
    return VGG_deep([1,1,2,2,2], num_classes=num_classes)