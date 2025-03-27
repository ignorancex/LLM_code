'''
Code is adapted from the monotone operator network repository on GitHub
https://github.com/locuslab/monotone_op_net
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NEMON(nn.Module):

    def __init__(self, in_dim, out_dim, m=0.05):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim, bias=True)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m
        self.d = nn.Parameter(torch.zeros(out_dim))

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_features),)

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.U(x),)

    def multiply(self, *z):
        #ATAz = self.A(z[0]) @ self.A.weight
        one = torch.ones(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        rowSums = torch.diag(torch.abs(self.A.weight) @ one).to(device = self.A.weight.device)
        #print(self.d)
        diagweights = torch.exp(self.d).to(device = self.A.weight.device)
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.A.weight.device)
        #print(torch.diag(diagweightsinverse))
        transformedA = torch.diag(diagweightsinverse) @ self.A.weight @ torch.diag(diagweights)
        transformedA.to(device = self.A.weight.device)
        z_out = self.m * z[0] + z[0] @ transformedA.T - z[0] @ rowSums
        #- ATAz + self.B(z[0]) - z[0] @ self.B.weight
        return (z_out,)

    def multiply_transpose(self, *g):
        #ATAg = self.A(g[0]) @ self.A.weight
        one = torch.ones(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        rowSums = torch.diag(torch.abs(self.A.weight) @ one).to(device = self.A.weight.device)
        diagweights = torch.exp(self.d).to(device = self.A.weight.device)
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.A.weight.device)
        transformedA = torch.diag(diagweightsinverse) @ self.A.weight @ torch.diag(diagweights)
        transformedA.to(device = self.A.weight.device)
        g_out = self.m * g[0] + g[0] @ transformedA - g[0] @ rowSums.T
        return (g_out,)

    def init_inverse(self, alpha, beta):
        I = torch.eye(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        one = torch.ones(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        rowSums = torch.diag(torch.abs(self.A.weight) @ one).to(device = self.A.weight.device)
        diagweights = torch.exp(self.d).to(device = self.A.weight.device)
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.A.weight.device)
        transformedA = torch.diag(diagweightsinverse) @ self.A.weight @ torch.diag(diagweights)
        transformedA.to(device = self.A.weight.device)
        W = self.m * I + transformedA - rowSums
        self.Winv = torch.inverse(alpha * I + beta * W)

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)


class NEMONReLU(nn.Module):
    def forward(self, *z):
        return tuple(F.relu(z_) for z_ in z)
        #return tuple(F.leaky_relu(z_,negative_slope=0.02) for z_ in z)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)
        #return tuple(0.98*(z_ > 0).type_as(z[0]) + 0.02*torch.ones_like(z_) for z_ in z)



def fft_to_complex_matrix(x):
    x_stacked = torch.stack((x, torch.flip(x, (4,))), dim=5).permute(2, 3, 0, 4, 1, 5)
    x_stacked[:, :, :, 0, :, 1] *= -1
    return x_stacked.reshape(-1, 2 * x.shape[0], 2 * x.shape[1])


def fft_to_complex_vector(x):
    return x.permute(2, 3, 0, 1, 4).reshape(-1, x.shape[0], x.shape[1] * 2)


def init_fft_conv(weight, hw):

    px, py = (weight.shape[2] - 1) // 2, (weight.shape[3] - 1) // 2
    kernel = torch.flip(weight, (2, 3))
    kernel = F.pad(F.pad(kernel, (0, hw[0] - weight.shape[2], 0, hw[1] - weight.shape[3])),
                   (0, py, 0, px), mode="circular")[:, :, py:, px:]
    return fft_to_complex_matrix(torch.rfft(kernel, 2, onesided=False))


def fft_conv(x, w_fft, transpose=False):

    x_fft = fft_to_complex_vector(torch.rfft(x, 2, onesided=False))
    wx_fft = x_fft.bmm(w_fft.transpose(1, 2)) if not transpose else x_fft.bmm(w_fft)
    wx_fft = wx_fft.view(x.shape[2], x.shape[3], wx_fft.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
    return torch.irfft(wx_fft, 2, onesided=False)



class NEMONSingleConv(nn.Module):
    """ Convolutional NEMON """

    def __init__(self, in_channels, out_channels, shp, kernel_size=5, m=0.05):
        super().__init__()
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        #self.a = nn.Parameter(torch.tensor(1.))
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def onepad(self, x):
        return F.pad(x, self.pad, mode="constant", value = 1.0)

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_channels, self.shp[0], self.shp[1]),)

    def forward(self, x, *z):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias) + self.multiply(*z)[0],)

    def bias(self, x):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias),)

    def multiply(self, *z):
        A = self.g * self.A.weight / self.A.weight.reshape(-1).norm()
        Az = F.conv2d(self.cpad(z[0]), A)
        z_out = (self.m - torch.max(torch.abs(A)) * self.out_channels) * z[0] + Az
        return (z_out,)

    def multiply_transpose(self, *g):
        A = self.g * self.A.weight / self.A.weight.reshape(-1).norm()
        ATg = self.uncpad(F.conv_transpose2d(self.cpad(g[0]), A))
        g_out = (self.m - torch.max(torch.abs(A)) * self.out_channels) * g[0] + ATg
        return (g_out,)

    def init_inverse(self, alpha, beta):
        A = self.g * self.A.weight / self.A.weight.reshape(-1).norm()
        Afft = init_fft_conv(A, self.shp)
        I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
                      device=Afft.device)[None, :, :]
        W = (self.m - torch.max(torch.abs(A)) * self.out_channels) * I + Afft
        self.Winv = torch.inverse(alpha * I + beta * W)

    def inverse(self, *z):
        return (fft_conv(z[0], self.Winv),)

    def inverse_transpose(self, *g):
        return (fft_conv(g[0], self.Winv, transpose=True),)


class NEMONBorderReLU(nn.Module):
    def __init__(self, border=1):
        super().__init__()
        self.border = border

    def forward(self, *z):
        zn = tuple(F.relu(z_, inplace=False) for z_ in z)
        result = []
        for i in range(len(zn)):
            out = torch.zeros_like(zn[i])
            out[:,:,self.border:-self.border, self.border:-self.border] = zn[i][:,:,self.border:-self.border, self.border:-self.border]
            result.append(out)
            
        return tuple(result)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)
