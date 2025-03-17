import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model.gcn import GCN
import scipy.sparse as sp

from torch_geometric.utils import to_scipy_sparse_matrix
from utils.convert import sparse_mx_to_torch_sparse_tensor,normalize_adj


class GRAND(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, args, use_bn=False):
        super(GRAND, self).__init__()
        self.mlp=MLP(nfeat, nhid,nclass,input_droprate, hidden_droprate).to(args.cuda_device)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.args=args

    def forward(self, x, edge_index):
        n = x.shape[0]

        adj= to_scipy_sparse_matrix(edge_index)
        adj=normalize_adj(adj+ sp.eye(adj.shape[0]))
        adj=sparse_mx_to_torch_sparse_tensor(adj).to(self.args.cuda_device)
        drop_rate = self.hidden_droprate
        drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
        if self.training:
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(self.args.cuda_device)
            x = masks * x
        else:
            x = x * (1. - drop_rate)
        x = self.propagate(x, adj, self.args.order)
        x=self.mlp(x)
        return x
    
    def propagate(self, x, adj, order):
        x = x
        y = x
        for i in range(order):
            x = torch.spmm(adj, x).detach_()
            y.add_(x)
        return y.div_(order+1.0).detach_()

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn =False):
        super(MLP, self).__init__()
        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)
        return x

class MLPLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features),requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
