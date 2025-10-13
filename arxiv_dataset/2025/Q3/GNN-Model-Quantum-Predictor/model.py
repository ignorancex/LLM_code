import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import Callable, List, Optional
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv


class QuantumGAT(nn.Module):
    def __init__(self, in_feats, hidden_dim, heads, mlp_units, num_residual=1, num_classes=4):
        """
        QuantumGAT model with GATConv and GCNConv layers.
        Args:
            in_feats (int): Number of input features.
            hidden_dim (int): Dimension of hidden layers.
            heads (int): Number of attention heads for GATConv. 
            mlp_units (List[int]): List of integers representing the number of units in each MLP layer.
            num_residual (int): Number of residual connections.
            num_classes (int): Number of output classes.
        """
        super(QuantumGAT, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(GATConv(in_feats, hidden_dim, heads=heads, concat=True))
        for _ in range(num_residual):
            self.conv.append(GCNConv(hidden_dim * heads, hidden_dim * heads))
        
        for out_dim in mlp_units:
            self.fcs.append(nn.Linear(last_dim, out_dim))
            last_dim = out_dim
        self.out = nn.Linear(last_dim, num_classes)
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.conv):
            if i == 0:
                x = conv(x, edge_index)
                F.leaky_relu(x)
            else:
                x_new = conv(x, edge_index)
                x = x + x_new
        
        
        x = global_mean_pool(x, batch)  

        for fc in self.fcs:
            x = F.leaky_relu(fc(x))

        return F.softmax(self.out(x))  



class QuantumGCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, mlp_units, num_residual=1, num_classes=4):
        """
        QuantumGAT model with GATConv and GCNConv layers.
        Args:
            in_feats (int): Number of input features.
            hidden_dim (int): Dimension of hidden layers.
            heads (int): Number of attention heads for GATConv. 
            mlp_units (List[int]): List of integers representing the number of units in each MLP layer.
            num_residual (int): Number of residual connections.
            num_classes (int): Number of output classes.
        """
        super(QuantumGAT, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(in_feats, hidden_dim))
        for _ in range(num_residual):
            self.conv.append(GCNConv(hidden_dim, hidden_dim))
        
        for out_dim in mlp_units:
            self.fcs.append(nn.Linear(last_dim, out_dim))
            last_dim = out_dim
        self.out = nn.Linear(last_dim, num_classes)
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.conv):
            if i == 0:
                x = conv(x, edge_index)
                F.leaky_relu(x)
            else:
                x_new = conv(x, edge_index)
                x = x + x_new
        
        
        x = global_mean_pool(x, batch)  

        for fc in self.fcs:
            x = F.leaky_relu(fc(x))

        return F.softmax(self.out(x))  



        
