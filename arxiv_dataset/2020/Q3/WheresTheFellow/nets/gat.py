"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch.nn as nn
from dgl.nn.pytorch.conv.gatconv import GATConv
import torch

class GAT(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_units, outputs, heads, activations, feat_drop,
                 attn_drop, negative_slope, residual=False):
        super(GAT, self).__init__()
        assert (len(heads) == num_layers), 'The length of the heads list must be the number of layers'
        assert (len(num_units) == num_layers), 'The length of the num_units list must be the number of layers'

        self.g = g
        self.layers = nn.ModuleList()
        # First layer
        print('L:{}   I({})   O({})'.format(0, in_dim, num_units[0]*heads[0]))
        self.layers.append(
            GATConv(in_dim, num_units[0], heads[0], feat_drop, attn_drop, negative_slope, False, activations[0]))
        # Hidden layers
        for layer_number in range(1, num_layers-1):
            print('L:{}   I({})   O({})'.format(layer_number, num_units[layer_number-1] * heads[layer_number-1], num_units[layer_number] * heads[layer_number]))
            self.layers.append(
                GATConv(num_units[layer_number-1] * heads[layer_number-1], num_units[layer_number], heads[layer_number],
                        feat_drop, attn_drop, negative_slope, residual, activations[layer_number]))
        # Output layer
        print('L:{}   I({})   O({})'.format(num_layers-1, num_units[-2] * heads[-2], num_units[0]*heads[0]))
        self.layers.append(GATConv(num_units[-2] * heads[-2], num_units[-1], outputs, feat_drop, attn_drop,
                                   negative_slope, residual, activations[-1]))

    def set_g(self, g):
        self.g = g

    def forward(self, inputs):
        h = inputs
        g = self.g.to(h.device)
        for layer_number in range(len(self.layers)-1):
            h = self.layers[layer_number](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)



