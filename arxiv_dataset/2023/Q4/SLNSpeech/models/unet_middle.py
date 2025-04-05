import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)


class Unet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.conv1x1 = create_conv(512, 8, 1, 0)

        self.up1 = Up(904, 512, bilinear)
        self.up2 = Up(1024, 256, bilinear)
        self.up3 = Up(512, 128, bilinear)
        self.up4 = Up(256, 64, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, visual_feat):
        # print(1, x.size())
        x1 = self.inc(x)
        # print(2, x1.size())
        x2 = self.down1(x1)
        # print(3, x2.size())
        x3 = self.down2(x2)
        # print(4, x3.size())
        x4 = self.down3(x3)
        # print(5, x4.size())
        x5 = self.down4(x4)
        # print(6, x5.size())

        visual_feat = self.conv1x1(visual_feat)
        visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
        visual_feat = visual_feat.repeat(1, 1, x5.shape[-2],
                                         x5.shape[-1])  # tile visual feature

        x = self.up1(x5, visual_feat)
        # print(5, x.size())
        x = self.up2(x, x4)
        # print(4, x.size())
        x = self.up3(x, x3)
        # print(3, x.size())
        x = self.up4(x, x2)
        # print(2, x.size())
        x = self.up5(x, x1)
        logits = self.outc(x)
        # print(1, logits.size())
        return logits, 1, 2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class Unet(nn.Module):
#     def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
#         super(Unet, self).__init__()
#
#         # construct unet structure
#         unet_block = UnetBlock(
#             ngf * 8, ngf * 8, input_nc=None,
#             submodule=None, innermost=True)
#         for i in range(num_downs - 5):
#             unet_block = UnetBlock(
#                 ngf * 8, ngf * 8, input_nc=None,
#                 submodule=unet_block, use_dropout=use_dropout)
#         unet_block = UnetBlock(
#             ngf * 4, ngf * 8, input_nc=None,
#             submodule=unet_block)
#         unet_block = UnetBlock(
#             ngf * 2, ngf * 4, input_nc=None,
#             submodule=unet_block)
#         unet_block = UnetBlock(
#             ngf, ngf * 2, input_nc=None,
#             submodule=unet_block)
#         unet_block = UnetBlock(
#             fc_dim, ngf, input_nc=1,
#             submodule=unet_block, outermost=True)
#
#         self.bn0 = nn.BatchNorm2d(1)
#         self.unet_block = unet_block
#
#     def forward(self, x):
#         x = self.bn0(x)
#         x = self.unet_block(x)
#         return x
#
#
# # Defines the submodule with skip connection.
# # X -------------------identity---------------------- X
# #   |-- downsampling -- |submodule| -- upsampling --|
# class UnetBlock(nn.Module):
#     def __init__(self, outer_nc, inner_input_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False,
#                  use_dropout=False, inner_output_nc=None, noskip=False):
#         super(UnetBlock, self).__init__()
#         self.outermost = outermost
#         self.noskip = noskip
#         self.innermost = innermost
#         use_bias = False
#         if input_nc is None:
#             input_nc = outer_nc
#         if innermost:
#             inner_output_nc = inner_input_nc
#         elif inner_output_nc is None:
#             inner_output_nc = 2 * inner_input_nc
#
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = nn.BatchNorm2d(inner_input_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = nn.BatchNorm2d(outer_nc)
#         upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=True)
#
#         if outermost:
#             downconv = nn.Conv2d(
#                 input_nc, inner_input_nc, kernel_size=4,
#                 stride=2, padding=1, bias=use_bias)
#             upconv = nn.Conv2d(
#                 inner_output_nc, outer_nc, kernel_size=3, padding=1)
#
#             down = [downconv]
#             up = [uprelu, upsample, upconv]
#             model = down + [submodule] + up
#         elif innermost:
#             downconv = nn.Conv2d(
#                 input_nc, inner_input_nc, kernel_size=4,
#                 stride=2, padding=1, bias=use_bias)
#             upconv = nn.Conv2d(
#                 inner_output_nc, outer_nc, kernel_size=3,
#                 padding=1, bias=use_bias)
#
#             down = [downrelu, downconv]
#             up = [uprelu, upsample, upconv, upnorm]
#             model = down + up
#         else:
#             downconv = nn.Conv2d(
#                 input_nc, inner_input_nc, kernel_size=4,
#                 stride=2, padding=1, bias=use_bias)
#             upconv = nn.Conv2d(
#                 inner_output_nc, outer_nc, kernel_size=3,
#                 padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upsample, upconv, upnorm]
#
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         if self.outermost or self.noskip:
#             # print(1, x.size())
#             # print(self.model(x).size())
#             return self.model(x)
#         # elif self.innermost is True:
#         #     print(3, x.size())
#         #     return self.model(torch.cat([x, x], 1))
#         else:
#             # print(2, x.size())
#             return torch.cat([x, self.model(x)], 1)


if __name__ == '__main__':

    model = Unet()
    dummy_input = torch.zeros(4, 1, 512, 320)
    output = model(dummy_input)
    print(output.shape)