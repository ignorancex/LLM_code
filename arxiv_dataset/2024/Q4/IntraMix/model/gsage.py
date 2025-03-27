import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, num_classes,hidden_num=128,dropout_rate1=0.9,dropout_rate2=0.9):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(num_node_features, hidden_num)
        self.sage2 = SAGEConv(hidden_num, num_classes)
        self.reg_params = self.sage1.parameters()
        self.non_reg_params = self.sage2.parameters()
        self.dropout_rate1=dropout_rate1
        self.dropout_rate2=dropout_rate2

    def forward(self, x,edge_index):
        x = F.dropout(x,self.dropout_rate1,training=self.training)
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,self.dropout_rate2,training=self.training)
        x = self.sage2(x, edge_index)
        return x

class GraphSAGE_BN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,num_layers,
                 dropout_rate1,dropout_rate2):
        super(GraphSAGE_BN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, adj_t):
        x=F.dropout(x, p=self.dropout_rate1, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate2, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x