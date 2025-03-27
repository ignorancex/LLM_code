import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.dropout = dropout
        self.conv2 = SAGEConv(nhid, nhid)
        self.conv2.aggr = 'mean'
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index): 
        x = self.conv1(x, edge_index)
        # x = self.transition(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # return self.fc(x)
        
        
        return self.fc(x)