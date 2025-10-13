import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv

"""
https://arxiv.org/abs/1810.00826
"""


class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass

        self.gin_convs = torch.nn.ModuleList()
        self.conv1 = GINConv(
            nn=Sequential(
                Linear(in_features=nfeat, out_features=nhid),
                # BatchNorm1d(num_features=nhid),
                ReLU(),
                Dropout(p=dropout),
                Linear(nhid, nhid),
                ReLU()
            ),
            # train_eps=True,
        )
        self.conv2 = GINConv(
            nn=Sequential(
                Linear(in_features=nhid, out_features=nhid),
                # BatchNorm1d(num_features=nhid),
                ReLU(),
                Dropout(p=dropout),
                Linear(nhid, nclass)
            ),
            # train_eps=True,
        )

    def forward(self, X, edge_idx):
        x = self.conv1(X, edge_idx)
        x = self.conv2(x, edge_idx)

        out = F.log_softmax(x, dim=-1)
        return out
