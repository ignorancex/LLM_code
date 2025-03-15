import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from src import MessagePassing
from torch_geometric.utils import softmax

from src.inits import glorot, zeros
import scipy.sparse as sp
import numpy as np

class SpGSDNEFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha, K, improved=False,
                 cached=False, bias=True, **kwargs):
        super(SpGSDNEFConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.K = K
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register('bias', None)

        self.beta = Parameter(torch.Tensor(1))

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
        self.beta.data.fill_(0.1)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        self.cached_num_edges = edge_index.size(1)
        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                        self.improved, x.dtype)
        self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        # out = x
        # x_norm = F.normalize(x, p=2, dim=-1)
        # for _ in range(self.K):
        #     out = self.alpha * self.propagate(edge_index, x=out, norm=norm, x_norm=x_norm, num_nodes=x.size(0)) + x
        # out = (1 - self.alpha) * out

        # return out

        x_norm = F.normalize(x, p=2, dim=-1)
        x = self.propagate(edge_index, x=x, norm=norm, x_norm=x_norm, num_nodes=x.size(0))

        return x

    def message(self, edge_index, x_j, norm, x_norm_i, x_norm_j, num_nodes):
        alpha = self.beta * (x_norm_j * x_norm_j).sum(dim=-1) + norm
        alpha[alpha < 0] = 0
        _, alpha = self.norm(edge_index, num_nodes, alpha)

        return x_j * alpha.view(-1, 1)
