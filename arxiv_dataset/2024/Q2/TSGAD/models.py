import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow
from graph import Graph
from st_graph_conv_block import ConvBlock
from tqdm import tqdm
from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img
    
class ConvEncoder2(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder2(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, h_dim=4, graph_args=None, split_seqs=True, eiw=True,
                 dropout=0.0, input_frames=12, conv_oper=None, act=None, headless=False, **kwargs):
        super(GraphEncoder, self).__init__()
        
        # Load graph
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        dec_1st_residual = kwargs.get('dec_1st_residual', None)
        
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        self.headless = headless
         # build networks
        num_node = self.graph.num_node
        self.fig_per_seq = 2
        if split_seqs:
            self.fig_per_seq = 1
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.kernel_size = kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.dropout = dropout
        self.act = get_act(act)  # Activation function for conv block. Defaults to ReLU

        self.in_channels = in_channels
        self.h_dim = h_dim

        arch_dict = {'enc_ch_fac': [4, 4, 4, 6, 6, 6, 8, 8, 4],
                     'enc_stride': [1, 1, 2, 1, 1, 3, 1, 1, 1],
                     'dec_ch_fac': [4, 8, 8, 6, 6, 6],
                     'dec_stride': [1, 3, 1, 1, 2, 1]}

        self.enc_ch_fac = arch_dict['enc_ch_fac']
        self.enc_stride = arch_dict['enc_stride']
        self.dec_ch_fac = arch_dict['dec_ch_fac']
        self.dec_stride = arch_dict['dec_stride']

        self.out_bn  = kwargs.get('out_bn', False)
        self.out_act = kwargs.get('out_act', False)
        self.out_res = kwargs.get('out_res', False)
        self.gen_ae(self.enc_ch_fac,
                    self.enc_stride,
                    self.dec_ch_fac,
                    self.dec_stride,
                    dec_1st_residual=dec_1st_residual)
        # self.lastact = nn.Sigmoid()
        downsample_factor = np.multiply.reduce(np.array(self.enc_stride))
        self.hidden_dim = (input_frames / downsample_factor) * num_node * h_dim * self.fig_per_seq
        self.hidden_dim *= self.enc_ch_fac[-1]

        # Edge weighting
        if eiw and (not conv_oper.startswith('sagc')):
            self.ei_enc = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_enc])
            # self.ei_dec = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_dec])
        else:
            self.ei_enc = [1] * len(self.st_gcn_enc)
            # self.ei_dec = [1] * len(self.st_gcn_dec)
    
    def forward(self, x):
        if self.fig_per_seq == 1:
            if len(x.size()) == 4:
                x = x.unsqueeze(4)
        # Return to (N*M, c, t, v) structure
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x.float())
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_enc, self.ei_enc):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        x = x.contiguous()
        x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        x_ref = x
        x_size = x.size()
        x = x.contiguous()
        x = x.view(N, -1)
        x = torch.sigmoid(x)
        return x, x_size, x_ref
    
    def gen_ae(self, enc_ch_fac, enc_stride, dec_ch_fac=None, dec_stride=None, symmetric=True, dec_1st_residual=True):
        if dec_ch_fac is not None or dec_stride is not None:
            symmetric = False
        # if symmetric:
        #     dec_ch_fac = enc_ch_fac[::-1]
        #     dec_stride = enc_stride[::-1]
        self.build_enc(enc_ch_fac, enc_stride)
        # self.build_dec(dec_ch_fac, dec_stride, dec_1st_residual=dec_1st_residual)

    def build_enc(self, enc_ch_fac, enc_stride):
        """
        Generate and encoder according to a series of dimension factors and strides
        """
        if len(enc_ch_fac) != len(enc_stride):
            raise Exception("Architecture error")

        enc_kwargs = [{'dropout': self.dropout, 'conv_oper': self.conv_oper, 'act': self.act,
                       'headless': self.headless} for _ in enc_ch_fac]
        enc_kwargs[0] = {'residual': False, **enc_kwargs[0]}
        enc_kwargs[-1] = {'out_act': False, **enc_kwargs[-1]}  # No Relu for final encoder layer
        st_gcn_enc = [ConvBlock(self.in_channels, enc_ch_fac[0] * self.h_dim, self.kernel_size, enc_stride[0],
                                **enc_kwargs[0])]
        for i in range(1, len(enc_ch_fac)):
            st_gcn_enc.append(
                ConvBlock(enc_ch_fac[i - 1] * self.h_dim, enc_ch_fac[i] * self.h_dim, self.kernel_size, enc_stride[i],
                          **enc_kwargs[i]))
        self.st_gcn_enc = nn.ModuleList(st_gcn_enc)
        
        
class GraphDecoder(nn.Module):
    def __init__(self, in_channels, eiw=True, h_dim =8, conv_oper=None, graph_args=None, split_seqs=True,
                 dropout=0.0, input_frames=12, act=None, headless=False, **kwargs):
        super(GraphDecoder, self).__init__()
        # self.x_size = x_size
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        dec_1st_residual = kwargs.get('dec_1st_residual', None)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        self.headless = headless
        self.h_dim = h_dim
        self.in_channels = in_channels
    
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        
        num_node = self.graph.num_node
        self.fig_per_seq = 2
        if split_seqs:
            self.fig_per_seq = 1
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = get_act(act)  # Activation function for conv block. Defaults to ReLU

        arch_dict = {'enc_ch_fac': [4, 4, 4, 6, 6, 6, 8, 8, 4],
                     'enc_stride': [1, 1, 2, 1, 1, 3, 1, 1, 1],
                     'dec_ch_fac': [4, 8, 8, 6, 6, 6],
                     'dec_stride': [1, 3, 1, 1, 2, 1]}

        self.enc_ch_fac = arch_dict['enc_ch_fac']
        self.enc_stride = arch_dict['enc_stride']
        self.dec_ch_fac = arch_dict['dec_ch_fac']
        self.dec_stride = arch_dict['dec_stride']

        self.out_bn  = kwargs.get('out_bn', False)
        self.out_act = kwargs.get('out_act', False)
        self.out_res = kwargs.get('out_res', False)
        self.gen_ae(self.enc_ch_fac,
                    self.enc_stride,
                    self.dec_ch_fac,
                    self.dec_stride,
                    dec_1st_residual=dec_1st_residual)

        
        # Edge weighting
        if eiw and (not conv_oper.startswith('sagc')):
            # self.ei_enc = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_enc]) # self.st_gcn_enc self.A
            self.ei_dec = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_dec])
        else:
            # self.ei_enc = [1] * len(self.st_gcn_enc)
            self.ei_dec = [1] * len(self.st_gcn_dec)
    
    def forward(self, z, x_size):
        # Decoding layers
        # x = z.view(self.x_size)
        # N, C, T, V, M = self.x_size
        N, C, T, V, M = x_size
        x = z.view(N, int(C/2), T, V, M)
        # x = z.view(x_size)
        # N, C, T, V, M = x_size
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, int(C/2), T, V)
        for ind, (layer_, importance) in enumerate(zip(self.st_gcn_dec, self.ei_dec)):
            if type(layer_) == ConvBlock:
                x, _ = layer_(x, self.A * importance)  # A graph convolution
            else:
                x = layer_(x)  # An upsampling layer

        x, _ = self.dec_final_gcn(x, self.A * self.ei_dec[-1])  # Final layer has no upsampling
        if self.fig_per_seq == 1:
            return x

        NM, c, t, v = x.size()
        x = x.view(N, M, c, t, v)
        mu_x = x.permute(0, 2, 3, 4, 1).contiguous()
        return mu_x
    
    def gen_ae(self, enc_ch_fac, enc_stride, dec_ch_fac=None, dec_stride=None, symmetric=True, dec_1st_residual=True):
        if dec_ch_fac is not None or dec_stride is not None:
            symmetric = False
        if symmetric:
            dec_ch_fac = enc_ch_fac[::-1]
            dec_stride = enc_stride[::-1]
        # self.build_enc(enc_ch_fac, enc_stride)
        self.build_dec(dec_ch_fac, dec_stride, dec_1st_residual=dec_1st_residual)
        
    def build_dec(self, dec_ch_fac, dec_stride, dec_1st_residual=True):
        if len(dec_ch_fac) != len(dec_stride):
            raise Exception("Architecture error")
        dec_kwargs = [{'dropout': self.dropout, 'conv_oper': self.conv_oper,
                       'act': self.act, 'headless': self.headless, } for _ in dec_ch_fac]
        dec_kwargs[1] = {'residual': dec_1st_residual, **dec_kwargs[1]}
        dec_kwargs += [{'residual': self.out_res, 'out_act': self.out_act, 'out_bn': self.out_bn, **dec_kwargs[0]}]
        st_gcn_dec = []
        for i in range(1, len(dec_ch_fac)):
            if dec_stride[i] != 1:
                st_gcn_dec.append(nn.Upsample(scale_factor=(dec_stride[i], 1), mode='bilinear'))

            st_gcn_dec.append(ConvBlock(dec_ch_fac[i - 1] * self.h_dim, dec_ch_fac[i] * self.h_dim, self.kernel_size, 1))

        # Add output layer back to in_channels w/o relu, bn or residuals
        if dec_kwargs[-1]['conv_oper'].startswith(('sagc')):
            dec_kwargs[-1]['conv_oper'] = 'gcn'
        self.dec_final_gcn = ConvBlock(dec_ch_fac[i] * self.h_dim, self.in_channels, self.kernel_size, 1,
                                       **(dec_kwargs[-1]))
        self.st_gcn_dec = nn.ModuleList(st_gcn_dec)
#######################################################


class VAE(nn.Module):
    def __init__(self, z_dim, device= 'cuda:0', use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(), alpha=1, gamma=1,
                 include_mutinfo=True, tcvae=False, conv=False, graph=False, mse=False, mss=False, type='convolutional', dropout=0.0, conv_oper=None, act=None, headless=False, in_channels=2, graph_args=None, split_seqs=True, input_frames=12, **kwargs):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.device = device
        self.mss = mss
        self.x_dist = dist.Bernoulli()
        self.graph = graph
        self.input_frames=input_frames
        self.mse = mse
        self.gamma = gamma
        self.alpha = alpha
        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        elif graph:
            self.encoder = GraphEncoder(in_channels=in_channels,
                                        input_frames=self.input_frames,
                                        h_dim=z_dim * self.q_dist.nparams, 
                                        graph_args=graph_args, 
                                        dropout=dropout, 
                                        conv_oper=conv_oper, 
                                        act=act, headless=headless, 
                                        split_seqs=split_seqs, 
                                        **kwargs)
            self.decoder = GraphDecoder(in_channels=in_channels,
                                        conv_oper=conv_oper,
                                        graph_args=graph_args,
                                        h_dim = z_dim,
                                        split_seqs=split_seqs,
                                        dropout=dropout,
                                        act=act,
                                        headless=headless,
                                        **kwargs)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    # def model_sample(self, batch_size=1):
    #     # sample from prior (value will be sampled by guide when computing the ELBO)
    #     prior_params = self._get_prior_params(batch_size)
    #     zs = self.prior_dist.sample(params=prior_params)
    #     # decode the latent code z
    #     x_params = self.decoder.forward(zs)
    #     return x_params
    def model_sample(self, latent_dim,batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        # prior_params = self._get_prior_params(batch_size)
        prior_params = torch.zeros((latent_dim))
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params
    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        # x = x.view(x.size(0), 1, 64, 64) no need for this as my data matches my model 
        # use the encoder to get the parameters used to define q(z|x)
        # z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        if self.graph:
            z_params, x_size, _  = self.encoder.forward(x)
            z_params = z_params.view(x.size(0), int(self.encoder.hidden_dim/self.q_dist.nparams), self.q_dist.nparams)
        else:
            x = x.view(x.size(0), 1, 64, 64)
            z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        if self.graph:
             return zs, z_params, x_size 
        else:
            return zs, z_params

    def decode(self, z, x_size=None):
        if self.graph:
            x_params = self.decoder.forward(z, x_size).view(z.size(0), 2, self.input_frames, 18) # hardcoded a bit N, C, T, V, M 
        else:
            x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        xs = self.x_dist.sample(params=x_params) # Why do you sample again? the output of the decoder is distribution parameters> Why?
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        if self.graph:
            zs, z_params, x_size = self.encode(x)
            xs, x_params = self.decode(zs, x_size=x_size)
        else:
            zs, z_params = self.encode(x)
            xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        
        if self.graph:
            x = x.view(batch_size, 2, self.input_frames, 18) 
        else:
            x = x.view(batch_size, 1, 64, 64)
        # prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        mse_loss = torch.mean((x-x_recon)**2)
        prior_params = torch.zeros((z_params.shape), device=self.device)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        
        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            if self.mse:
                elbo = self.gamma * elbo - self.alpha * mse_loss
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        if self.graph:
            _logqz = self.q_dist.log_density(
                zs.view(batch_size, 1, int(self.encoder.hidden_dim/self.q_dist.nparams)),
                z_params.view(1, batch_size, int(self.encoder.hidden_dim/self.q_dist.nparams), self.q_dist.nparams)
            )
        else:
            _logqz = self.q_dist.log_density(
                zs.view(batch_size, 1, self.z_dim),
                z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
            )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        if self.mse:
            modified_elbo = self.gamma * modified_elbo - self.alpha * mse_loss
            
        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


# def anneal_kl(args, vae, iteration):
#     if args.dataset == 'shapes':
#         warmup_iter = 7000
#     elif args.dataset == 'faces':
#         warmup_iter = 2500

#     if args.lambda_anneal:
#         vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
#     else:
#         vae.lamb = 0
#     if args.beta_anneal:
#         vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
#     else:
#         vae.beta = args.beta



def get_act(act_type):
    if act_type is None:
        return nn.ReLU(inplace=True)
    if act_type.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type.lower() == 'mish':
        return Mish()


class Mish(nn.Module):
    """
    Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))