# code is from: https://github.com/jianzhangcs/ISTA-Net-PyTorch/blob/master/Train_MRI_CS_ISTA_Net_plus.py

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
from method.DeCoLearn import  inputDataDict, absolute_helper
# Define ISTA-Net-plus Block

class BasicBlock(torch.nn.Module):
    def __init__(self, channels): # Channel is 2 for fmri and simulated data.
        super(BasicBlock, self).__init__()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.channels = channels
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]).to(device))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]).to(device))

        self.conv_D = BasicConv2d(in_channels=self.channels, out_channels=32, using_relu=False).to(device)

        self.conv1_forward = BasicConv2d(in_channels=32, out_channels=32).to(device)
        self.conv2_forward = BasicConv2d(in_channels=32, out_channels=32, using_relu=False).to(device)
        self.conv1_backward = BasicConv2d(in_channels=32, out_channels=32).to(device)
        self.conv2_backward = BasicConv2d(in_channels=32, out_channels=32, using_relu=False).to(device)

        #self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 32, 3, 3).to(device)))
        self.conv_G = BasicConv2d(in_channels=32, out_channels=2, using_relu=False).to(device)

        #self.basic_conv2 = BasicConv2d(32, self.conv1_forward, padding=1)
        #self.basic_conv3 = BasicConv2d(32, self.conv2_forward, padding=1)
        #self.basic_conv4 = BasicConv2d(32, self.conv1_backward, padding=1)
        #self.basic_conv5 = BasicConv2d(32, self.conv2_backward, padding=1)
        #self.basic_conv6 = BasicConv2d(32, self.conv_G, padding=1)
        #self.basic_conv6 = BasicConv2d(32, self.conv_D, padding=1)

    #def forward(self, x, y, ftran, fmult):
    def forward(self, XPSY):
        x, P, S, y, ftran, fmult = XPSY.getData()

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
        #x = self.lambda_step(XPSY)
        x = x - self.lambda_step * ftran(fmult(x) - y)
        x_input = x

        if self.channels == 2:
            x_input = torch.view_as_real(x_input)
            x_input = rearrange(x_input, 'b w h c -> b c w h')

        x_D = self.conv_D(x_input)

        x = self.conv1_forward(x_D)

        x_forward = self.conv2_forward(x)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = self.conv1_backward(x)

        x_backward = self.conv2_backward(x)

        x_G= self.conv_G(x_backward)

        x_pred = x_input + x_G

        x = self.conv1_backward(x_forward)

        x_D_est = self.conv2_backward(x)

        symloss = x_D_est - x_D

        if self.channels == 2:
            x_pred = rearrange(x_pred, 'b c w h -> b w h c')
            x_pred = x_pred[..., 0] + x_pred[..., 1] * 1j

        return [x_pred, symloss]

class BasicConv2d(nn.Module):
    def __init__(self, in_channels=2, out_channels=32, using_relu=True):
        super(BasicConv2d, self).__init__()
        self.using_relu = using_relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=None)
        self.conv.weight = nn.Parameter(init.xavier_normal_(self.conv.weight.data))

    def forward(self, x):
        x = self.conv(x)
        if self.using_relu == True:
            output = F.relu(x, inplace=True)
        else:
            output = x
        return output

# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, channels, LayerNo=9):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock(channels))

        self.fcs = nn.ModuleList(onelayer)

    #def forward(self, x, y, ftran, fmult):
    def forward(self, XPSY):

        _, P, S, y, ftran, fmult = XPSY.getData()

        #x = x

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](XPSY)
            XPSY = inputDataDict(x, P, S, y, module_name="ISTANET")
            #def __init__(self, x_init, P, S, y, module_name=None):

            layers_sym.append(layer_sym)

        x_final = x

        x_final = (torch.view_as_real(x_final)).permute([0, 3, 1, 2]).contiguous()

        return [x_final, layers_sym]

'''

# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.channels = channels
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]).to(device))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]).to(device))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, self.channels, 3, 3)))

        #self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        #self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        #self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        #self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        #self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 32, 3, 3)))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, self.channels, 3, 3).to(device)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3).to(device)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3).to(device)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3).to(device)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3).to(device)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 32, 3, 3).to(device)))


    #def forward(self, x, y, ftran, fmult):
    def forward(self, XPSY):
        x, P, S, y, ftran, fmult = XPSY.getData()

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

        #print(f"x_input.shape: {x_input.shape}")

        x_D = F.conv2d(x_input, self.conv_D, padding=1)


        #print(f"x_D.shape: {x_D.shape}")

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)

        #print(f"x.shape: {x.shape}")

        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        #print(f"x_forward.shape: {x_forward.shape}")

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

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, channels, LayerNo=9):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock(channels))

        self.fcs = nn.ModuleList(onelayer)

    #def forward(self, x, y, ftran, fmult):
    def forward(self, XPSY):

        _, P, S, y, ftran, fmult = XPSY.getData()

        #x = x

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](XPSY)
            XPSY = inputDataDict(x, P, S, y, module_name="ISTANET")
            layers_sym.append(layer_sym)

        x_final = x

        x_final = (torch.view_as_real(x_final)).permute([0, 3, 1, 2]).contiguous()

        return [x_final, layers_sym]
       
'''
