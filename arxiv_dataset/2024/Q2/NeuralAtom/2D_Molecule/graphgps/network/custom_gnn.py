import math

import torch
import torch.nn.functional as F
import torch_geometric.graphgym.models.head
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from einops import einsum, rearrange, reduce
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gcnii_conv_layer import GCN2ConvLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.mlp_layer import MLPLayer
from graphgps.layer.neural_atom import Exchanging, Porjecting
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch


class CustomGNN(torch.nn.Module):
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

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        self.model_type = cfg.gnn.layer_type

        layers = []
        multi_NA = []
        NA_trans = []
        
        num_out_nodes = cfg.gvm.avg_nodes

        num_NAs = [] # for incre number of NAs
        for _ in range(cfg.gnn.layers_mp):
            num_NAs.append(max(math.ceil(cfg.gvm.pool_ratio * num_out_nodes), 1))
            num_NAs = num_NAs[::-1]
            
        for layer_idx in range(cfg.gnn.layers_mp):
            
            if cfg.gvm.na_order == "desc":
                num_out_nodes = max(math.ceil(cfg.gvm.pool_ratio * num_out_nodes), 1)
            elif cfg.gvm.na_order == "fixed":
                num_out_nodes = math.ceil(cfg.gvm.pool_ratio * cfg.gvm.avg_nodes)
            elif cfg.gvm.na_order == "incre":
                num_out_nodes = num_NAs[layer_idx]
                
            if self.model_type == "gcn":
                layer = conv_model(dim_in, dim_in)
            else:
                layer = conv_model(
                    dim_in, dim_in, dropout=cfg.gnn.dropout, residual=cfg.gnn.residual
                )
            layers.append(layer)

            multi_NA.append(
                Porjecting(
                    dim_in,
                    num_heads=cfg.gvm.n_pool_heads,
                    num_seeds=num_out_nodes,
                    Conv=None,
                    layer_norm=True,
                )
            )

            NA_trans.append(
                Exchanging(
                    dim_in,
                    dim_in,
                    num_heads=cfg.gvm.n_pool_heads,
                    Conv=None,
                    layer_norm=True,
                )
            )

        self.gnn_layers = torch.nn.Sequential(*layers)
        self.multi_NA_layers = torch.nn.Sequential(*multi_NA)
        self.NA_trans_layers = torch.nn.Sequential(*NA_trans)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == "gatedgcnconv":
            return GatedGCNLayer
        elif model_type == "gineconv":
            return GINEConvLayer
        elif model_type == "gcniiconv":
            return GCN2ConvLayer
        elif model_type == "mlp":
            return MLPLayer
        elif model_type == "gcn":
            return pyg_nn.GCNConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        batch = self.encoder(batch)
        batch.x0 = batch.x  # gcniiconv needs x0 for each layer

        in_x = batch.x
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)

        for idx, layer in enumerate(self.gnn_layers):
            # * add message from virtual nodes to graph nodes
            if self.model_type == "gcn":
                batch.x = layer(batch.x, batch.edge_index)
            else:
                batch = layer(batch)
            vns_emb = self.get_NAs_emb(batch, idx)
            if cfg.dataset.name == "PCQM4Mv2Contact-shuffle":
                batch.x = (
                    batch.x + 0.01 * vns_emb
                )  # in_x and 0.2 * vns_emb for LP with GINE, GCNII and GCN only
            else:
                batch.x = batch.x + vns_emb

            # dropout for every layer
            # h = F.dropout(h, self.drop_ratio, training=self.training)

            # residual for every layer
            # batch.x = in_x + batch.x

        batch = self.post_mp(batch)
        return batch

    def get_NAs_emb(self, batch, idx):
        batch_x, ori_mask = to_dense_batch(batch.x, batch.batch)
        mask = (~ori_mask).unsqueeze(1).to(dtype=batch.x.dtype) * -1e9
        # * S for neural atom allocation matrix
        vns_emb, S = self.multi_NA_layers[idx](batch_x, None, mask)
        vns_emb = self.NA_trans_layers[idx](vns_emb, None, None)[0]
        h = reduce(
            einsum(
                rearrange(S, "(b h) c n -> h b c n", h=cfg.gvm.n_pool_heads),
                vns_emb,
                "h b c n, b c d -> h b n d",
            ),
            "h b n d -> b n d",
            "mean",
        )[ori_mask]
        return h


register_network("custom_gnn", CustomGNN)
