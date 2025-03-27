import torch
import torch.nn as nn
import torch.nn.functional as F
from .guided_filter_pytorch.guided_filter import FastGuidedFilter
from .LaplacianPyramid import Lap_Pyramid
from .CorrectionNet import U_Net_4, U_Net_3,deconv_Block
#import thop

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)

def build_lr_net(norm=AdaptiveNorm, layer=2, width=24):#lr = low resolution
    layers = [
        nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1,layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(width),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(width,  3, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    return net

class FusionNet(nn.Module):
    # end-to-end mef model
    def __init__(self, depth,radius=1, eps=1e-4, is_guided=True):
        super(FusionNet, self).__init__()
        self.lr = build_lr_net(layer=depth)
        self.is_guided = is_guided
        if is_guided:
            self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        w_lr = self.lr(x_lr)

        if self.is_guided:
            w_hr = self.gf(x_lr, w_lr, x_hr)
        else:
            w_hr = F.upsample(w_lr, x_hr.size()[2:], mode='bilinear')

        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + 1e-8) / torch.sum((w_hr + 1e-8), dim=0)

        o_hr = torch.sum(w_hr * x_hr, dim=0, keepdim=True).clamp(0, 1)

        return o_hr, w_hr

#Multi_Scale_model
class FCNet(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, num_high):
        super().__init__()
        self.F_net_1 = FusionNet(depth=4)
        self.F_net_2 = FusionNet(depth = 3)
        self.F_net_3 =FusionNet(depth = 2)
        self.F_net_4 =FusionNet(depth = 1)
        self.C_net_1 = U_Net_4(24)#channel 24
        self.C_net_2 = U_Net_4(16)#channel 16
        self.C_net_3 = U_Net_3(16)#channel 16
        self.C_net_4 = U_Net_3(16)#channel 16
        self.decov = deconv_Block(in_channels=3, out_channels=3)

        self.lp = Lap_Pyramid(num_high)
        self.num_high = num_high

    def forward(self, img):
        #lp decomposition
        list = self.lp.pyramid_decom(img=img)
        #fusion on the 4th layer
        image_lr = nn.functional.interpolate(list[3], size=(list[3].shape[2] // 2, list[3].shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_1,_ = self.F_net_1(image_lr, list[3])
        #correction
        out_1 = self.C_net_1(F_1)
        #upsampling
        U_1 = self.decov(out_1,list[2])
        U_1 = list[2] + U_1
        # fusion on the 3th layer
        image_lr = nn.functional.interpolate(U_1, size=(U_1.shape[2] // 2, U_1.shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_2, _ =  self.F_net_2(image_lr, U_1)
        # correction
        out_2 = self.C_net_2(F_2)
        # upsampling
        U_2 = self.decov(out_2, list[1])
        U_2 = list[1] + U_2
        # fusion on the 2th layer
        image_lr = nn.functional.interpolate(U_2, size=(U_2.shape[2] // 2, U_2.shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_3, _ = self.F_net_3(image_lr, U_2)
        # correction
        out_3 = self.C_net_3(F_3)
        # upsampling
        U_3 = self.decov(out_3, list[0])
        U_3 = list[0] + U_3
        # fusion on the 1th layer
        image_lr = nn.functional.interpolate(U_3, size=(U_3.shape[2] // 2, U_3.shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_4, _ = self.F_net_4(image_lr, U_3)
        # correction
        out_4 = self.C_net_4(F_4)

        return out_1, out_2, out_3, out_4

if __name__ == "__main__":
    #x_0 = torch.rand(size=(2,3,3))
    x_1 = torch.rand(size=(7, 3, 75, 113)).cuda()
    x_2 = torch.rand(size=(7, 3, 150, 226)).cuda()
    x_3 = torch.rand(size=(7, 3, 300, 452)).cuda()
    x_4 = torch.rand(size=(7, 3, 600, 903)).cuda()
    x_5 = torch.rand(size=(7, 3, 523, 173)).cuda()

    net2 = FCNet(num_high=3).cuda()
    out1 , out2, out3, out4 = net2(x_5)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)