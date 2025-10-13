# code is from: https://github.com/jianzhangcs/ISTA-Net-PyTorch/blob/master/Train_MRI_CS_ISTA_Net_plus.py

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()

        self.channels = channels
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        '''
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, self.channels, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 32, 3, 3)))
        '''
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, self.channels, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 32, 3, 3)))

    def forward(self, x, y, ftran, fmult):
        """
        :param x: undersampled_image, shape: batch, 2, width, height; dtype: float32
        :param y: undersampled measurement, shape: batch, coils, width, height, 2; dtype: float32
        :param ftran: function
        :param fmult: function
        """

        #print(f"[istanetpluspmri] x.shape: {x.shape} / x.dtype: {x.dtype}")
        #print(f"[istanetpluspmri] y.shape: {y.shape} / y.dtype: {y.dtype}")
        if len(x.shape) != 3:
            x = x.permute([0, 2, 3, 1]).contiguous()
            x = torch.view_as_complex(x)
        if len(y.shape) != 4:
            y = torch.view_as_complex(y)

        #print(f"[istanetpluspmri] ftran(fmult(x) - y).shape: {ftran(fmult(x) - y).shape} / ftran(fmult(x) - y).dtype: {ftran(fmult(x) - y).dtype}")

        x = x - self.lambda_step * ftran(fmult(x) - y)
        #print(f"[istanetpluspmri] x.shape: {x.shape} / x.dtype: {x.dtype}")
        x_input = x

        if self.channels == 2:
            x_input = torch.view_as_real(x_input)
            x_input = rearrange(x_input, 'b w h c -> b c w h')

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        if self.channels == 2:
            x_pred = rearrange(x_pred, 'b c w h -> b w h c')
            x_pred = x_pred[..., 0] + x_pred[..., 1] * 1j

        return [x_pred, symloss]


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, channels, LayerNo=9):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock(channels))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, y, ftran, fmult):

        x = x

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, y, ftran, fmult)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
