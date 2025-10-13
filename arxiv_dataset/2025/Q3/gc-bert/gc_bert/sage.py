import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads=8):
        super().__init__()

        self.gc1 = SAGEConv(in_channels=nfeat, out_channels=nhid, aggr='add', normalize=True)
        self.gc2 = SAGEConv(in_channels=nhid, out_channels=nclass, aggr='add', normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_idx):
        x = self.gc1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_idx)
        return F.log_softmax(x, dim=1)
