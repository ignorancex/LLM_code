from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from hive.algorithm.gnn.gat.model import GAT


class VStreamDRLSModel(Module):
    def __init__(self, layer_1_in, layer_1_out, layer_2_out, n_heads, dropout = 0.1, alpha=0.2, activation = nn.ReLU()):
        super(VStreamDRLSModel, self).__init__()
        self.layer2_weight = Parameter(torch.zeros(size=(layer_1_out, layer_2_out)))
        nn.init.xavier_uniform_(self.layer2_weight)

        self.attention_weights_init = Parameter(torch.zeros(size=(layer_1_in, layer_1_out)))
        nn.init.xavier_uniform_(self.attention_weights_init)

        self.self_attention = GAT(in_features=layer_1_out, n_hidden=layer_1_out, n_out=layer_1_out, n_heads=n_heads, dropout=dropout, alpha=alpha)

        self.activation = activation


    def forward(self, adj_list, adj_norm, features):
        attention_weights = self.attention_weights_init
        for t, adj in enumerate(adj_list):
            attention_weights = self.self_attention(attention_weights, adj)

        h1 = self.activation(torch.mm(adj_norm, torch.mm(features, attention_weights)))
        embs = torch.mm(adj_norm, torch.mm(h1, self.layer2_weight))
        return embs
