# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from numpy import fft
import numpy as np

## U-Net utils
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, do not change (H,W)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # only double H,w
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] # (B,C,H,W)
        diffX = x2.size()[3] - x1.size()[3]
        # print('pad info:',diffX,diffY)  # debug

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

class TinyUNet(nn.Module):
    def __init__(self, n_channels, n_out_channel, bilinear=True):
        super(TinyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_out_channel = n_out_channel
        self.bilinear = bilinear
        factor = 2 if bilinear else 1   # when use biliner method, pre reduce the channel 

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))   # C: 128 + 128, max channel 256
        self.up2 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_out_channel))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # xc = x1 + x     # residual path
        logits = self.outc(x)
        return logits

    
## 1d wiener
def wiener_deconv(input, kernel, SNR_db=0):       
    ''' 
    Winner filter with given blur kernel
    ---
    input: M长序列，可能包含周边padding
    kernel: 补全至M的卷积核
    '''
    K = 1 / (np.power(10, SNR_db / 10))
    input_fft = fft.fft(input)
    kernel_fft = fft.fft(kernel)
    kernel_fft_1 = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = fft.ifft(input_fft * kernel_fft_1)
    # result = np.abs(fft.fftshift(result))
    return result

def pre_deconv(input_samples, M_users, L_symbols, h_coff, SNR_db=0):
    ''' 
    Restore all samples(MxL,) from misaligned y(MxL+M-1,)
    ---
    input_samples: M*(L+1)-1长复数序列
    M_users: 用户数量
    L_symbols: 每个用户发送数据长度
    h_coff: 信道增益，需要已知
    SNR_db: 信道质量，需要已知

    return: M*L长序列, (M,L)二维数组
    '''
    
    input_samples = input_samples.flatten()
    if len(input_samples) != M_users*L_symbols+M_users-1:
        raise RuntimeError('The shape of input is:{}, not equals to M*L+M-1'.format(str(len(input_samples))))
    kernel_k = np.array([1.]*M_users)
    k_pad = np.zeros_like(input_samples)
    k_pad[0:M_users] = 1.
    out = wiener_deconv(input_samples,k_pad,SNR_db)  # 输出应该是复数序列
    out = out.flatten()[:(L_symbols*M_users)]  # 这一步输出的是Hs+K^-1 n
    out_mat = out.reshape(L_symbols,M_users).T
    h_coff = h_coff.reshape(-1,1)
    # print(h_coff)   # debug
    hh_expand = np.tile(h_coff,(1,out_mat.shape[1]))    # 每行是hi
    out_mat = out_mat/hh_expand # 输出s + H^-1 K^-1 n
    out_seq = out_mat.T.flatten()
    return out_seq, out_mat

# usage:
# PIL read image
# y_img = y_img/255.
# y_samples = y_img[:,:,0] +1j*y_img[:,:,1] 
# y_samples = y_samples[:L_symbols*M_users].reshape(L_symbols,M_users).T    # 从序列读入
# y_samples = (y_samples-np.min(y_samples))/(np.max(y_samples)-np.min(y_samples)) # 归一化
# _,s_deconv = pre_deconv(y_samples, M_users, L_symbols, h_coff, SNR_db=0)
# s_deconv <- array[real,imag]
# s_estimate = model(s_deconv)
# s_sum_est = torch.sum(s_estimate,axis=1)

if __name__ == '__main__':
    import time
    model = TinyUNet(2,2,bilinear=True).cuda()
    model.train()
    criteria = nn.MSELoss()
    input = torch.randn((100,2,20,1024)).cuda()
    target = torch.randn((100,2,20,1024)).cuda()
    start_time = time.time()
    for i in range(100):
        out = model(input[i,:,:,:].unsqueeze(0))
        loss = criteria(out,target[i,:,:,:].unsqueeze(0))
        loss.backward()
    end_time = time.time()-start_time
    print('Input shape:',input.size(),'Output shape:',out.size(),'\nUse {} s'.format(end_time))