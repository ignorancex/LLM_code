import torch
from torch.nn import Conv2d, Conv3d, MaxPool2d, MaxPool3d
from torch.nn import BatchNorm2d, BatchNorm3d
from torch.nn import AdaptiveAvgPool2d, AdaptiveAvgPool3d
from lib.models.BaseModel import BaseModel
from lib.blocks.MedicDeepLabv3PlusBlocks import *
from lib.blocks.BasicBlocks import *

class MedicDeepLabv3Plus(BaseModel):
    """
    """
    params = ["modalities", "n_classes", "first_filters", "dim"]
    def __init__(self, modalities, n_classes, first_filters=32, dim="3D"):
        """Incorporates a skip connection in the upper-most level.
           Replaces teh last convolution with a RatLesNetv2 block.
        """
        super(MedicDeepLabv3Plus, self).__init__()
        
        self.interpol_mode = "bilinear" if dim == "2D" else "trilinear"
        # Xception
        self.xception = Xception_DeepLabv3Plus_nskips(modalities, first_filters,
                n_classes, dim)

        self.aspp_L = DeepLabv3_ASPP(first_filters*64,
                first_filters*8, [6, 12, 18], dim)

        self.project = DeepLabv3_Head(first_filters*8, n_classes,
                dim, last_layer=False)

        if dim == "2D":
            Conv = Conv2d
        elif dim == "3D":
            Conv = Conv3d

        self.conv1 = Conv(first_filters*16, first_filters*8, 3, padding=1)
        self.block1 = DeepLabv3Plus_ResNetBlock(first_filters*8, dim)
        self.convout64 = Depthwise_Sep_Conv(first_filters*8, n_classes, Conv)

        self.conv2 = Conv(first_filters*8+first_filters*4, first_filters*4, 3, padding=1)
        self.block2 = DeepLabv3Plus_ResNetBlock(first_filters*4, dim)
        self.convout128 = Depthwise_Sep_Conv(first_filters*4, n_classes, Conv)

        self.conv3 = Conv(first_filters*4+first_filters, first_filters*2, 3, padding=1)
        self.block3 = DeepLabv3Plus_ResNetBlock(first_filters*2, dim)

        self.convout256 = Depthwise_Sep_Conv(first_filters*2, n_classes, Conv)
        self.conv_final = Conv(first_filters*2, n_classes, 3, padding=1)

        self.cat = Cat()
        self.interpolate = Interpolate()
        self.softmax = Softmax()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x[0]
        out, skip1, skip2, skip3 = self.xception(x)
        out = self.aspp_L(out)
        out = self.project(out)
        out = self.interpolate(out, size=list(skip3.size()[2:]),
                mode=self.interpol_mode, align_corners=False)
        out = self.cat([out, skip3], dim=1)

        out = self.conv1(out) # Half # of filters
        out = self.block1(out)

        # Attention mechanism 1
        out64_pre = self.convout64(out)
        out64 = self.interpolate(out64_pre, size=[18, 64, 64],
                mode=self.interpol_mode, align_corners=False)

        out64_imp = torch.sigmoid(torch.mean(out64_pre, dim=1).view(x.shape[0], 1, out64_pre.shape[2], out64_pre.shape[3], out64_pre.shape[4]))
        out = out * out64_imp

        out = self.interpolate(out, size=skip2.shape[2:],
                mode=self.interpol_mode, align_corners=False)
        out = self.cat([out, skip2], dim=1)

        out = self.conv2(out) # Half # of filters
        out = self.block2(out)

        # Attention mechanism 2
        out128_pre = self.convout128(out)
        out128 = self.interpolate(out128_pre, size=[18, 128, 128],
                mode=self.interpol_mode, align_corners=False)

        out128_imp = torch.sigmoid(torch.mean(out128_pre, dim=1).view(x.shape[0], 1, out128_pre.shape[2], out128_pre.shape[3], out128_pre.shape[4]))
        out = out * out128_imp

        out = self.interpolate(out, size=skip1.shape[2:],
                mode=self.interpol_mode, align_corners=False)
        out = self.cat([out, skip1], dim=1)

        out = self.conv3(out) # Half # of filters
        out = self.block3(out)

        # Attention mechanism 3
        out256_pre = self.convout256(out)
        out256_imp = torch.sigmoid(torch.mean(out256_pre, dim=1).view(x.shape[0], 1, out256_pre.shape[2], out256_pre.shape[3], out256_pre.shape[4]))
        out = out * out256_imp

        # block + last_conv
        out = self.conv_final(out)
        out = self.softmax(out, dim=1)

        return (out, self.softmax(out128, dim=1), self.softmax(out64, dim=1))

class Xception_DeepLabv3Plus_nskips(BaseModel):
    """Xception used by a new version of DeepLabv3Plus that has 3 skip
       connections.
    """
    params = ["modalities", "first_filters", "n_classes", "dim"]
    def __init__(self, modalities, first_filters=32, n_classes=20, dim="2D"):
        super(Xception_DeepLabv3Plus_nskips, self).__init__()

        # Originally, first_filters = 32
        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
            MaxPool = MaxPool2d
            AvgPool = AdaptiveAvgPool2d
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d
            MaxPool = MaxPool3d
            AvgPool = AdaptiveAvgPool3d

        nfi = first_filters
        nf2 = int(22.75*nfi) # Originall: 728
        layers1 = []

        ### Entry Flow
        self.xception_part1 = nn.Sequential(
                Conv(modalities, nfi, kernel_size=3, stride=1, padding=1),
                #DeepLabv3Plus_ResNetBlock(nfi, dim),
                BN(nfi),
                ReLU(),
                )

        self.xception_part2 = nn.Sequential(
                Conv(nfi, nfi*2, kernel_size=3, stride=2, padding=1),
                BN(nfi*2),
                ReLU()
                )

        # Module 1
        self.xception_part3 = Sep_Conv_DeepLabv3Plus(
            filters=[nfi*2, nfi*4, nfi*4, nfi*4], stride=2, dilation=1,
            skip_con="conv", dim=dim, return_skip=True)

        # Module 2
        self.xception_part4 = Sep_Conv_DeepLabv3Plus(
            filters=[nfi*4, nfi*8, nfi*8, nfi*8], stride=2, dilation=1,
            skip_con="conv", dim=dim, return_skip=True)

        layers2 = []
        # Module 3
        layers2.append(Sep_Conv_DeepLabv3Plus(
            filters=[nfi*8, nf2, nf2, nf2], stride=2, dilation=1,
            skip_con="conv", dim=dim))

        for i in range(8):
            layers2.append(Sep_Conv_DeepLabv3Plus(
                filters=[nf2, nf2, nf2, nf2], stride=1, dilation=1,
                skip_con="sum", dim=dim))

        # Exit flow
        layers2.append(Sep_Conv_DeepLabv3Plus(
            filters=[nf2, nf2, nfi*32, nfi*32], stride=1, dilation=1,
            skip_con="conv", dim=dim))
        
        layers2.append(Sep_Conv_DeepLabv3Plus(
            filters=[nfi*32, nfi*48, nfi*48, nfi*64], stride=1, dilation=2,
            skip_con="none", dim=dim))

        self.xception_part5 = nn.Sequential(*layers2)

    def forward(self, x):
        skip1 = self.xception_part1(x)
        x1 = self.xception_part2(skip1)
        x1, skip2 = self.xception_part3(x1)
        x1, skip3 = self.xception_part4(x1)
        x2 = self.xception_part5(x1)

        return x2, skip1, skip2, skip3

