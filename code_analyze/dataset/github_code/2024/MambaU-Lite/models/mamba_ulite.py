import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vssblock import *
from models.selective_kernel import *
from models.ipca import *
from models.att_schemes import *



class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x
    
class Encoder_Axial(nn.Module):
      def __init__(self, in_c, out_c, mixer_kernel = (7,7)):
          super().__init__()
          self.adw = AxialDW(in_c, mixer_kernel = mixer_kernel )
          self.bn = nn.BatchNorm2d(in_c)
          self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
          self.down = nn.MaxPool2d((2,2))
          self.act = nn.ReLU()

      def forward(self, x):
          x = self.adw(x)
          skip = self.act(self.bn(x))
          x = self.pw(skip)
          x = self.down(x)
          return x, skip


class MaxAvg(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv33 = nn.Conv2d(in_c+in_c, in_c, kernel_size = 3, padding = 'same')
        self.sigmoid = nn.Sigmoid()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.maxp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        ori = x
        x_avg = self.avgp(x)
        x_max = self.maxp(x)
        x = torch.cat([x_avg, x_max], dim = 1)
        x = self.conv33(x)
        x = self.sigmoid(x)
        x = x * ori

        return x
    
    
class MambaSplit(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.block = VSSBlock(hidden_dim = in_c // 2)
        self.dw33 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        ori = x
        x = self.dw33(x)

        x_1, x_2 = torch.chunk(x,2, dim = 1)

        x1 = x_1.permute(0, 2, 3, 1)
        x1 = self.block(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.scale*x_1 + x1

        x2 = x_2.permute(0, 2, 3, 1)
        x2 = self.block(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.scale*x_2 + x2

        x = torch.cat([x1, x2], dim=1)
        x = self.act(self.ins_norm(x))
        return x
    
class Encoder_Mamba(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mamba = MambaSplit(in_c)
        self.maxavg = MaxAvg(in_c)
        self.adw = AxialDW(in_c, mixer_kernel = (3,3))
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.ReLU()
        self.down = nn.MaxPool2d((2,2))

    def forward(self, x):
        x1 = self.mamba(x)
        x2 = self.maxavg(x)
        x = x1+x2
        x = self.adw(x)
        skip = self.act(self.bn(x))
        x = self.pw(skip)
        x = self.down(x)

        return x, skip
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.att = Attention_block(F_g = in_c, F_l = skip_c, F_int = skip_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.adw = AxialDW(out_c, mixer_kernel = (7,7))

    def forward(self, x, skip):
        x = self.up(x)
        x = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.adw(x)
        x = self.pw2(x)

        return x
    
    
class ULite(nn.Module):
    def __init__(self):
        super().__init__()

        self.pw_in = nn.Conv2d(3, 16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1 ,L=32)
        self.pw1 = nn.Conv2d(16, 1, kernel_size=1)
        self.pw2 = nn.Conv2d(32, 1, kernel_size=1)
        self.pw3 = nn.Conv2d(64, 1, kernel_size=1)
        self.pw4 = nn.Conv2d(128, 1, kernel_size=1)

        """Encoder"""
        self.e1 = Encoder_Mamba(16, 32)
        self.e2 = Encoder_Mamba(32, 64)
        # self.e1m = Encoder_Mamba(8, 16)
        # self.e1a = Encoder_Axial(8, 16)
        # self.e2m = Encoder_Mamba(16, 32)
        # self.e2a = Encoder_Axial(16, 32)
        self.e3m = Encoder_Mamba(32, 64)
        self.e3a = Encoder_Axial(32, 64)
        self.e4m = Encoder_Mamba(64, 128)
        self.e4a = Encoder_Axial(64, 128)

        """Skip connection"""
        self.s1 = CBAM(gate_channels = 16)
        self.s2 = CBAM(gate_channels = 32)
        self.s3 = CBAM(gate_channels = 64)
        self.s4 = CBAM(gate_channels = 128)

        """Bottle Neck"""
        #self.b5 = SKUnit(512, 512, 512, M=2, G=16, r=2, stride=1, L=32)
        self.b5 = BottleneckPCAPSA(256)


        """Decoder"""
        self.d4 = DecoderBlock(256, 128, 128)
        self.d3 = DecoderBlock(128, 64, 64)
        self.d2 = DecoderBlock(64, 32, 32)
        self.d1 = DecoderBlock(32, 16, 16)
        self.conv_out = nn.Conv2d(4, 1, kernel_size=1)
        # self.out = OutBlock(3, 1)

    def forward(self, x):
        """Encoder"""
        H, W = x.size()[2:]
        x = self.pw_in(x)
        x = self.sk_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)

        xm, xa = torch.chunk(x, 2, dim = 1)

        # xm, skip1m = self.e1m(xm)
        # xa, skip1a = self.e1a(xa)
        # skip1 = torch.cat([skip1m, skip1a], dim = 1)

        # xm, skip2m = self.e2m(xm)
        # xa, skip2a = self.e2a(xa)
        # skip2 = torch.cat([skip2m, skip2a], dim = 1)

        xm, skip3m = self.e3m(xm)
        xa, skip3a = self.e3a(xa)
        skip3 = torch.cat([skip3m, skip3a], dim = 1)

        xm, skip4m = self.e4m(xm)
        xa, skip4a = self.e4a(xa)
        skip4 = torch.cat([skip4m, skip4a], dim = 1)

        """Skip connection"""
        skip1 = self.s1(skip1)

        skip2 = self.s2(skip2)

        skip3 = self.s3(skip3)

        skip4 = self.s4(skip4)


        """BottleNeck"""
        x = torch.cat([xm, xa], dim = 1)
        x = self.b5(x)

        """Decoder"""

        x4 = self.d4(x, skip4)
        x3 = self.d3(x4, skip3)
        x2 = self.d2(x3, skip2)
        x1 = self.d1(x2, skip1)

        x_in4 = self.pw4(x4)
        x_in3 = self.pw3(x3)
        x_in2 = self.pw2(x2)
        x_in1 = self.pw1(x1)

        x_in4 = F.interpolate(x_in4, size=(H, W), mode="bilinear", align_corners=False)
        x_in3 = F.interpolate(x_in3, size=(H, W), mode="bilinear", align_corners=False)
        x_in2 = F.interpolate(x_in2, size=(H, W), mode="bilinear", align_corners=False)
        x_in1 = F.interpolate(x_in1, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([x_in4, x_in3, x_in2, x_in1], dim=1)
        x = self.conv_out(x)
        return x
