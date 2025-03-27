import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer


class BaseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, layer_name='gcnconv'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        if layer_name == 'gcnconv':
            self.model = pyg_nn.GCNConv(self.dim_in, self.dim_out)
        if layer_name == 'sageconv':
            self.model = pyg_nn.SAGEConv(self.dim_in, self.dim_out)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        return batch
