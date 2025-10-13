import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import Module
from bigg.common.pytorch_util import *

# --------------------------------------------------------------
# Adapted functions for BiGG-E originally from BiGG (Google Research):
# https://github.com/google-research/google-research/blob/c097eb6c850370c850eb7a90ab8a22fd2e1c730a/ugsl/input_layer.py#L103
# Functions adapted: TreeLSTMCell, BinaryTreeLSTMCell
# Copyright (c) Google LLC
# Licensed under the Apache License 2.0
# --------------------------------------------------------------

class TreeLSTMCellWt(nn.Module):
    def __init__(self, arity, latent_dim, latent_dim_wt = None, use_mlp = False):
        super(TreeLSTMCellWt, self).__init__()
        self.arity = arity
        if latent_dim_wt is None:
            self.latent_dim = arity * latent_dim
        
        else:
            self.latent_dim = latent_dim + latent_dim_wt
        
        if use_mlp:
            self.mlp_i = MLP(self.latent_dim, [2 * self.latent_dim, latent_dim], act_last='sigmoid')
            self.mlp_o = MLP(self.latent_dim, [2 * self.latent_dim, latent_dim], act_last='sigmoid')
            self.mlp_u = MLP(self.latent_dim, [2 * self.latent_dim, latent_dim], act_last='tanh')
        
        else:
            self.mlp_i = nn.Sequential(nn.Linear(self.latent_dim, latent_dim), nn.Sigmoid())
            self.mlp_o =  nn.Sequential(nn.Linear(self.latent_dim, latent_dim), nn.Sigmoid())
            self.mlp_u =  nn.Sequential(nn.Linear(self.latent_dim, latent_dim), nn.Tanh())
        
        f_list = []
        for _ in range(arity):
            if use_mlp:
                mlp_f = MLP(self.latent_dim, [2 * self.latent_dim, latent_dim], act_last='sigmoid')
            else:
                mlp_f =  nn.Sequential(nn.Linear(self.latent_dim, latent_dim), nn.Sigmoid())
            
            f_list.append(mlp_f)
        self.f_list = nn.ModuleList(f_list)

    def forward(self, list_h_mat, list_c_mat):
        assert len(list_c_mat) == self.arity == len(list_h_mat)
        h_mat = torch.cat(list_h_mat, dim=-1)
        assert h_mat.shape[-1] == self.latent_dim

        i_j = self.mlp_i(h_mat)

        f_sum = 0
        for i in range(self.arity):
            f = self.f_list[i](h_mat)
            f_sum = f_sum + f * list_c_mat[i]

        o_j = self.mlp_o(h_mat)
        u_j = self.mlp_u(h_mat)
        c_j = i_j * u_j + f_sum
        h_j = o_j * torch.tanh(c_j)
        return h_j, c_j

class BinaryTreeLSTMCellWt(TreeLSTMCellWt):
    def __init__(self, latent_dim, latent_dim_wt = None, use_mlp = False):
        super(BinaryTreeLSTMCellWt, self).__init__(2, latent_dim, latent_dim_wt, use_mlp)

    def forward(self, lch_state, rch_state):
        list_h_mat, list_c_mat = zip(lch_state, rch_state)
        return super(BinaryTreeLSTMCellWt, self).forward(list_h_mat, list_c_mat)



































