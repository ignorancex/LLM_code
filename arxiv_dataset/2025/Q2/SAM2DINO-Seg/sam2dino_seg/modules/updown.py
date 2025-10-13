import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
class SmallKernelUpsample(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.conv_trans2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_trans1(x))  # (1,256,40,40)
        x = self.conv_trans2(x)             # (1,256,44,44)
        return x

class interpolate_upsample(nn.Module):
    def __init__(self,up_size):
        super().__init__()
        self.up_size = up_size

    def forward(self, x):
        # 使用双线性插值法进行上采样
        # (1,1024,37,37)->(1,1024,88,88)
        x = F.interpolate(x, size=(self.up_size, self.up_size), mode='bilinear', align_corners=False)
        return x

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class  mixconvpool(nn.Module):
    def __init__(self,in_planes,out_planes,down_size,kernel_size,stride):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0)
        self.down_size = down_size
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 使用双线性插值法进行上采样
        # (1,1024,37,37)->(1,1024,11,11)
        x = self.conv(x)
        x = F.interpolate(x, size=(self.down_size, self.down_size), mode='bilinear', align_corners=False)
        x = self.bn(x)
        x = self.relu(x)
        return x



if __name__ == '__main__':
    input = torch.randn(12, 1024, 37, 37)  #B C H W

    block = SmallKernelUpsample(1024,1024)
    block1 = interpolate_upsample(11)
    block2 = Down_wt(1024,1024)
    block3 = mixconvpool(1024,1024,11,3,3)
    block4 = mixconvpool(1024, 1024, 22, 2, 2)

    print(input.size())

    output = block(input)
    output1 = block1(input)
    output2 = block2(input)
    output3 = block3(input)
    output4 = block4(input)
    print(output.size())
    print(output1.size())
    print(output2.size())
    print(output3.size())
    print(output4.size())