import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GATConv
    
class GAT(nn.Module):
    def __init__(self, num_feature, num_label,hidden_num=16,nb_heads=8,nb_heads2=1,dropout_rate1=0.6,dropout_rate2=0.6,alpha=0.2):
        super(GAT,self).__init__()
        self.GAT1 = GATConv(num_feature, hidden_num, heads = nb_heads, dropout = dropout_rate1)
        self.GAT2 = GATConv(hidden_num*nb_heads, num_label, heads=nb_heads2,concat=False,dropout = dropout_rate2)  
        self.dropout_rate1=dropout_rate1
        self.dropout_rate2=dropout_rate2
        self.alpha=alpha
        
    def forward(self, x,edge_index):
        x = F.dropout(x,self.dropout_rate1,training=self.training)
        x = self.GAT1(x, edge_index)
        x = F.elu(x,alpha=self.alpha)
        x = F.dropout(x,self.dropout_rate2,training=self.training)
        x = self.GAT2(x, edge_index)
        return x
    
class GAT_BN(torch.nn.Module):
    def __init__(self, num_feature, num_label,hidden_num=16,layer=3,nb_heads=8,nb_heads2=1,dropout_rate1=0.6,dropout_rate2=0.6,alpha=0.2):
        super(GAT_BN, self).__init__()
        self.nlayer = layer
        self.convs = torch.nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(num_feature,hidden_num,heads=nb_heads,dropout=dropout_rate1))
        self.bns.append(nn.BatchNorm1d(hidden_num*nb_heads))
        for _ in range(self.nlayer - 2):
            self.convs.append(GATConv(hidden_num * nb_heads,hidden_num,heads=nb_heads,dropout=dropout_rate2))
            self.bns.append(nn.BatchNorm1d(hidden_num * nb_heads))
        self.convs.append(GATConv(hidden_num * nb_heads,num_label,heads=nb_heads2,concat=False,dropout=dropout_rate2))
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.alpha=alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x=F.dropout(x, p=self.dropout_rate1, training=self.training)
        for i in range(self.nlayer-1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x,alpha=self.alpha)
            x = F.dropout(x, p=self.dropout_rate2, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x