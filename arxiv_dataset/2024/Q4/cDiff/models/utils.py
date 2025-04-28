import math
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


#####
# Utility functions and classes
#####

def zscore(y, dim=None):
    ybar = y.mean(dim=dim, keepdim=True)
    ysd = y.std(dim=dim, keepdim=True)
    z = (y - ybar) / ysd
    return z, (ybar.squeeze(), ysd.squeeze())


def maxscore(y, dim=None):
    ybar = y.mean(dim=dim, keepdim=True)
    y = y - ybar
    ymax, _ = y.abs().max(dim=dim, keepdim=True)
    z = y / ymax
    return z, (ybar.squeeze(), ymax.squeeze())


import math
import torch
import numpy as np


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1


def sample_gaussian(shape):
    return torch.randn(*shape)


def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')


######################################################################################################################
########################################### Conditional MLPS #########################################################
######################################################################################################################

import einops
import math
import torch
from torch import nn
import numpy as np


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network which can be generated with different
    activation functions with and without spectral normalization of the weights
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            device: str = 'cuda'
    ):
        super(MLPNetwork, self).__init__()
        self.network_type = "mlp"
        # define number of variables in an input sequence
        self.input_dim = input_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = output_dim
        # set up the network
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        for i in range(1, self.num_hidden_layers):
            self.layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Mish()
            ])
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self._device = device
        self.layers.to(self._device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()


class ScoreNetwork(nn.Module):
    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            time_embed_dim: int,
            cond_dim: int,
            cond_mask_prob: float,
            num_hidden_layers: int = 1,
            output_dim=1,
            device: str = 'cuda',
            cond_conditional: bool = True
    ):
        super(ScoreNetwork, self).__init__()
        # Gaussian random feature embedding layer for time
        # self.embed = GaussianFourierProjection(time_embed_dim).to(device)
        self.embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        ).to(device)
        self.time_embed_dim = time_embed_dim
        self.cond_mask_prob = cond_mask_prob
        self.cond_conditional = cond_conditional
        if self.cond_conditional:
            input_dim = self.time_embed_dim + x_dim + cond_dim
        else:
            input_dim = self.time_embed_dim + x_dim
            # set up the network
        self.layers = MLPNetwork(
            input_dim,
            hidden_dim,
            num_hidden_layers,
            output_dim,
            device
        ).to(device)

        # build the activation layer
        self.act = nn.Mish()
        self.device = device
        self.sigma = None
        self.training = True

    def forward(self, x, cond, sigma, uncond=False):
        # Obtain the feature embedding for t
        if len(sigma.shape) == 0:
            sigma = einops.rearrange(sigma, ' -> 1')
            sigma = sigma.unsqueeze(1)
        elif len(sigma.shape) == 1:
            sigma = sigma.unsqueeze(1)
        embed = self.embed(sigma)
        embed.squeeze_(1)
        if embed.shape[0] != x.shape[0]:
            embed = einops.repeat(embed, '1 d -> (1 b) d', b=x.shape[0])
        # during training randomly mask out the cond
        # to train the conditional model with classifier-free guidance wen need
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2
        if self.training and cond is not None:
            cond = self.mask_cond(cond)
        # we want to use unconditional sampling during classifier free guidance
        if uncond:
            cond = torch.zeros_like(cond)  # cond
        if self.cond_conditional:
            x = torch.cat([x, cond, embed], dim=-1)
        else:
            x = torch.cat([x, embed], dim=-1)
        x = self.layers(x)
        return x  # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, d),
                                              device=cond.device) * self.cond_mask_prob)  # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()



########################################################################################################################
######################################### Basic Layer Component ########################################################
########################################################################################################################


class BasicConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=True, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return F.elu(x)


class ResidualConv1D(nn.Module):
    # note: simple residual conv block that has as many out_channels as in_channels
    def __init__(self, in_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, bias=True, **kwargs)

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.conv(x)
        return F.elu(identity + x)


class Inception1D(nn.Module):
    def __init__(self,
                 in_dim: int,
                 n_filters: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.n_filters = n_filters

        common_kwargs = {'padding': 'same',
                         'padding_mode': 'replicate'}

        self.conv1 = BasicConv1D(in_channels=in_dim, out_channels=n_filters,
                                 kernel_size=1, **common_kwargs)

        self.conv3_1 = BasicConv1D(in_channels=in_dim, out_channels=n_filters,
                                   kernel_size=1,
                                   **common_kwargs)
        self.conv3_2 = BasicConv1D(in_channels=n_filters, out_channels=n_filters,
                                   kernel_size=3,
                                   **common_kwargs)

        self.conv5_1 = BasicConv1D(in_channels=in_dim, out_channels=n_filters,
                                   kernel_size=1,
                                   **common_kwargs)
        self.conv5_2 = BasicConv1D(in_channels=n_filters, out_channels=n_filters,
                                   kernel_size=5,
                                   **common_kwargs)

    def forward(self, x):
        # input is (batch_size, n_channels, seq_length)
        branch1x = self.conv1(x)
        branch3x = self.conv3_1(x)
        branch3x = self.conv3_2(branch3x)
        branch5x = self.conv5_1(x)
        branch5x = self.conv5_2(branch5x)
        outputs = [branch1x, branch3x, branch5x]
        return torch.cat(outputs, 1)


class Inception1DResidualBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_filters: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_filters = n_filters

        common_kwargs = {'padding': 'same',
                         'padding_mode': 'replicate'}

        self.conv1 = BasicConv1D(in_channels=in_dim, out_channels=n_filters,
                                 kernel_size=1, **common_kwargs)

        self.conv3_1 = BasicConv1D(in_channels=in_dim, out_channels=n_filters,
                                   kernel_size=1, **common_kwargs)
        self.conv3_2 = BasicConv1D(in_channels=n_filters, out_channels=n_filters,
                                   kernel_size=3, **common_kwargs)

        self.conv9_1 = BasicConv1D(in_channels=in_dim, out_channels=n_filters,
                                   kernel_size=1, **common_kwargs)
        self.conv9_2 = BasicConv1D(in_channels=n_filters, out_channels=n_filters,
                                   kernel_size=3, **common_kwargs)
        self.conv9_3 = BasicConv1D(in_channels=n_filters, out_channels=n_filters,
                                   kernel_size=3, **common_kwargs)

        self.conv_1_res = nn.Conv1d(3 * n_filters, out_dim, kernel_size=1, bias=True)
        self.conv_1_x = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        # input is (batch_size, n_channels, seq_length)
        identity = x
        branch1x = self.conv1(x)
        branch3x = self.conv3_1(x)
        branch3x = self.conv3_2(branch3x)
        branch9x = self.conv9_1(x)
        branch9x = self.conv9_2(branch9x)
        branch9x = self.conv9_3(branch9x)
        outputs = torch.cat([branch1x, branch3x, branch9x], 1)
        outputs = self.conv_1_res(outputs)
        if self.in_dim != self.out_dim:
            identity = self.conv_1_x(identity)
        outputs = identity + outputs
        return F.leaky_relu(outputs, negative_slope=0.1)


# This layer is useful anytime we need to hand off between an LSTM layer and a conv layer.
# LSTM layers want data as (batch_size, seq_length, seq_depth) when batch_first=True.
# conv layers want data as (batch_size, seq_depth, seq_length).
class SwapLengthDepth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        if X.dim() == 2:
            X = X.transpose(0, 1)
        elif X.dim() == 3:
            X = X.transpose(1, 2)
        else:
            raise Exception("Can only swap length and depth for a tensor of dimension 2 (unbatched) or 3 (batched).")
        return X
