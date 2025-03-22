import torch
import torch.nn.functional as F
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.egnn_layer import EGNNConv


class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc, bias):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(
            torch.Tensor(
                nc,
            )
        )
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, bias)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias


class EGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        self.beta = cfg.egnn.beta
        self.c_min = cfg.egnn.c_min
        self.bias_SReLU = cfg.egnn.bias_SReLU
        self.dropout = cfg.egnn.dropout
        self.output_dropout = cfg.egnn.output_dropout

        conv_model = EGNNConv
        self.model_type = cfg.gnn.layer_type
        layers = []
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.reg_params = []

        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in, dim_in, c_max=cfg.egnn.c_max, bias=False))
            self.layers_activation.append(SReLU(dim_in, self.bias_SReLU))
            self.reg_params.append(layers[-1].weight)
            self.layers_bn.append(torch.nn.BatchNorm1d(dim_in))
        self.gnn_layers = torch.nn.Sequential(*layers)

        self.srelu_params = list(self.layers_activation[:-1].parameters())

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
        self.non_reg_params = list(self.encoder.parameters()) + list(
            self.post_mp.parameters()
        )
        self.non_reg_params += list(self.layers_bn[:-1].parameters())

    def forward(self, batch):
        # batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        batch.x = F.relu(batch.x)

        original_x = batch.x

        for idx, layer in enumerate(self.gnn_layers):
            batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            batch.x = layer(
                x=batch.x,
                edge_index=batch.edge_index,
                x_0=original_x,
                beta=self.beta,
                residual_weight=residual_weight,
                edge_weight=None,  # batch.edge_attr,
            )
            batch.x = self.layers_bn[idx](batch.x)
            batch.x = self.layers_activation[idx](batch.x)

        # * task specific head
        batch.x = F.dropout(batch.x, p=self.output_dropout, training=self.training)
        batch = self.post_mp(batch)

        return batch


register_network("egnn", EGNN)
