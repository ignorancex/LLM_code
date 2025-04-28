'''SSIM in PyTorch.

The source code is adopted from:
https://github.com/Po-Hsun-Su/pytorch-ssim


Reference:
[1] Wang Z, Bovik A C, Sheikh H R, et al.
    Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

def uniform(window_size,sigma):
    uniform_tensor = torch.ones(window_size)
    return uniform_tensor / uniform_tensor.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = uniform(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None, drop=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2/2

    L = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    C = (2*torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    S = (sigma12 + C3) / (torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq) + C3)

    if drop == "L":
        ssim_map = C*S
    elif drop == "C":
        ssim_map = L*S
    elif drop == "S":
        ssim_map = L*C
    elif drop == "LC":
        ssim_map = S
    else:
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LocalGlance(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3, drop=None, sigma=1.5, channel=1):
        super(LocalGlance, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.stride = stride
        self.window = create_window(window_size, self.channel, sigma)
        self.drop = drop
        self.sigma = sigma


    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,2,h,w]) - 2表示复数的实部和虚部
        """
        # 计算幅值
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride, drop=self.drop)       


def ssim(img1, img2, window_size = 11, size_average = True, sigma=1.5):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, sigma)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)








