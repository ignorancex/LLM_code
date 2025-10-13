import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor, masked_select_nnz
from torch_geometric.nn import GCNConv
from torch.nn import LayerNorm
from typing import Final

class DropAdj(torch.nn.Module):
    doscale: Final[bool]
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = masked_select_nnz(adj, mask, layout="coo") 
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, edge_drop):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.adjdrop = DropAdj(edge_drop)

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
            self.lins.append(LayerNorm(hidden_channels))

        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.lins.append(LayerNorm(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
                self.lins.append(LayerNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     
    def forward(self, x, adj_t):
        adj_t_drop = self.adjdrop(adj_t)
        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv, lin in zip(self.convs[:-1], self.lins):
            x = conv(x, adj_t_drop)
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t_drop)
        
        return x
    
class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)