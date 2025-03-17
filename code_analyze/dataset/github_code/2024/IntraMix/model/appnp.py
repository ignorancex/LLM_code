import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import APPNP


class APPNP_Net(torch.nn.Module):
    def __init__(self,num_node_features, num_classes,args,hidden_num=128,dropout_rate1=0.9,dropout_rate2=0.005):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(num_node_features, hidden_num)
        self.lin2 = Linear(hidden_num, num_classes)
        self.prop1 = APPNP(args.K, args.appnp_alpha)
        self.dropout1 = dropout_rate1
        self.dropout2 = dropout_rate2

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x
    
class APPNP_BN(torch.nn.Module):
    def __init__(self,num_node_features, num_classes,args,num_layers=3,hidden_num=128,dropout_rate1=0.9,dropout_rate2=0.005):
        super(APPNP_BN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(Linear(num_node_features, hidden_num))
        self.bns = self.bns.append(nn.BatchNorm1d(hidden_num))
        for i in range(self.num_layers - 2):
            self.convs.append(Linear(hidden_num, hidden_num))
            self.bns.append(nn.BatchNorm1d(hidden_num))
        self.convs.append(Linear(hidden_num, num_classes))
        self.prop1 = APPNP(args.K, args.appnp_alpha)
        self.dropout1 = dropout_rate1
        self.dropout2 = dropout_rate2

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout1, training=self.training)
        for i in range(self.num_layers - 1):
            x = self.convs[i](x)
            x = self.prop1(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.convs[-1](x)
        return x