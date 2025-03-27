import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self,num_node_features, num_classes,hidden_num=128,dropout_rate1=0.9,dropout_rate2=0.005):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_num)
        self.conv2 = GCNConv(hidden_num, num_classes)
        self.dropout_rate1=dropout_rate1
        self.dropout_rate2=dropout_rate2
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self,x,edge_index):
        x = F.dropout(x,self.dropout_rate1,training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x,self.dropout_rate2,training=self.training)
        x = self.conv2(x, edge_index)
        return x  
    
class GCNNet(nn.Module):
    def __init__(self, num_node_features, num_classes, dropout_rate1,dropout_rate2,hidden_num=256, num_layers=3):
        super(GCNNet, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_num,cached=True))
        self.bns.append(nn.BatchNorm1d(hidden_num))
        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden_num, hidden_num,cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_num))
        self.convs.append(GCNConv(hidden_num, num_classes,cached=True))
        self.dropout_rate1=dropout_rate1
        self.dropout_rate2=dropout_rate2

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x,adj_t):
        x = F.dropout(x, p=self.dropout_rate1, training=self.training)
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate2, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x