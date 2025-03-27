import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body1 = GCN_Body(nfeat,nhid,dropout)
        self.body2 = GCN_Body(nhid,nclass,dropout)
        # self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

        # self.weights_init()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            # torch.nn.init.uniform_(m.weight.data, a=-stdv, b=stdv)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                # torch.nn.init.uniform_(m.bias.data, a=-stdv, b=stdv)

    def forward(self, x, edge_index):

        x = F.relu(self.body1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.body2(x, edge_index)
        # x = self.fc(x)

        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)/10
        x = torch.div(torch.sub(x, mean), std)

        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        # normalization brings nan if noise is added
        self.gc1 = GCNConv(nfeat, nhid, normalize=True, add_self_loops=True)  # False

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x

