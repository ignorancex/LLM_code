import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from einops import einsum, rearrange, reduce
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.mlp_mixer_block import MLPMixer
from graphgps.layer.neural_atom import Exchanging, Porjecting
from performer_pytorch import SelfAttention
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch


class GVMLayer(nn.Module):
    """Local MPNN + VN + Trasnformer layer."""

    def __init__(
        self,
        dim_h,
        num_out_nodes,
        pool_num_heads,
        layers_mp,
        #  mixer_dim, mixer_depth, token_exp_fac, mixer_dropout,
        global_model_type,
        local_gnn_type,
        num_heads,
        pna_degrees=None,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        bigbird_cfg=None,
    ):
        super().__init__()

        self.num_out_nodes = num_out_nodes
        self.pool_num_heads = pool_num_heads

        # self.mixer_dim = mixer_dim
        # self.mixer_depth = mixer_depth
        # self.token_exp_fac = token_exp_fac
        # self.mixer_dropout = mixer_dropout

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe

        # * Local message-passing model.
        # * ========================================
        if local_gnn_type == "None":
            self.local_model = None
        elif local_gnn_type == "GENConv":
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == "GINE":
            gin_nn = nn.Sequential(
                Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h)
            )
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)

            # ! Now Only support GINE for multiple Layer GNN
            layers = []
            bns = []
            dropouts = []
            for _ in range(layers_mp):
                layers.append(
                    pygnn.GINEConv(
                        nn.Sequential(
                            Linear_pyg(dim_h, dim_h),
                            nn.ReLU(),
                            Linear_pyg(dim_h, dim_h),
                        )
                    )
                )
                bns.append(nn.BatchNorm1d(dim_h))
                dropouts.append(nn.Dropout(dropout))

            self.local_layers = torch.nn.Sequential(*layers)
            self.local_bns = torch.nn.Sequential(*bns)
            self.local_dropouts = torch.nn.Sequential(*dropouts)

        elif local_gnn_type == "GAT":
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == "PNA":
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ["mean", "max", "sum"]
            scalers = ["identity"]
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=16,  # dim_h,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == "CustomGatedGCN":
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                equivstable_pe=equivstable_pe,
            )

            layers = []
            bns = []
            dropouts = []
            for _ in range(layers_mp):
                layers.append(
                    GatedGCNLayer(
                        dim_h,
                        dim_h,
                        dropout=dropout,
                        residual=True,
                        equivstable_pe=equivstable_pe,
                    )
                )
                bns.append(nn.BatchNorm1d(dim_h))
                dropouts.append(nn.Dropout(dropout))

            self.local_layers = torch.nn.Sequential(*layers)
            self.local_bns = torch.nn.Sequential(*bns)
            self.local_dropouts = torch.nn.Sequential(*dropouts)

        elif local_gnn_type == "GAT":
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # * Local message-passing model.
        # * ========================================
        # Global attention transformer-style model.
        if global_model_type == "None":
            self.self_attn = None
        elif global_model_type == "Transformer":
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True
            )
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == "Performer":
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False
            )
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(
                f"Unsupported global x-former model: " f"{global_model_type}"
            )
        self.global_model_type = global_model_type

        # * Graph Virtual Nodes Pooling (adopt from GMT)
        # * ========================================
        self.vn_pool_pma = PMA(
            dim_h, self.pool_num_heads, self.num_out_nodes, Conv=None, layer_norm=True
        )

        # self.vn_pool_sab = SAB(
        #     dim_h, dim_h, self.pool_num_heads, Conv=None, layer_norm=True
        # )
        # * Multiple VN Mixer
        # * ========================================
        # self.vn_mixer = MLPMixer(
        #     in_dim=dim_h, dim=self.mixer_dim, depth=self.mixer_depth, expansion_factor_token=self.token_exp_fac, dropout=self.mixer_dropout)

        # * Normalization for MPNN and VN representations.
        # * ========================================
        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        # self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)

        # * Feed Forward block.
        # * ========================================
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # * Local MPNN with edge attributes.
        # * ========================================
        if self.local_model is not None:
            # self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            # if self.local_gnn_type == 'CustomGatedGCN':
            #     es_data = None
            #     if self.equivstable_pe:
            #         es_data = batch.pe_EquivStableLapPE
            #     local_out = self.local_model(Batch(batch=batch,
            #                                        x=h,
            #                                        edge_index=batch.edge_index,
            #                                        edge_attr=batch.edge_attr,
            #                                        pe_EquivStableLapPE=es_data))
            #     # GatedGCN does residual connection and dropout internally.
            #     h_local = local_out.x
            #     batch.edge_attr = local_out.edge_attr
            # else:
            #     if self.equivstable_pe:
            #         h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
            #                                    batch.pe_EquivStableLapPE)
            #     else:
            #         h_local = self.local_model(
            #             h, batch.edge_index, batch.edge_attr)
            #     h_local = self.dropout_local(h_local)
            h_local = h
            for gnn_layer, bn, dropout in zip(
                self.local_layers, self.local_bns, self.local_dropouts
            ):
                if self.local_gnn_type == "CustomGatedGCN":
                    es_data = None
                    if self.equivstable_pe:
                        es_data = batch.pe_EquivStableLapPE
                    local_out = self.local_model(
                        Batch(
                            batch=batch,
                            x=h,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            pe_EquivStableLapPE=es_data,
                        )
                    )
                    # GatedGCN does residual connection and dropout internally.
                    h_local = local_out.x
                    batch.edge_attr = local_out.edge_attr
                else:
                    h_local = gnn_layer(h_local, batch.edge_index, batch.edge_attr)

                h_local = dropout(h_local)
                h_local = bn(h_local)

            h_local = h_in1 + h_local  # Residual connection.
            # if self.layer_norm:
            #     h_local = self.norm1_local(h_local, batch.batch)
            # if self.batch_norm:
            #     h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # * Graph pooling for multiple VN
        # * ========================================
        # ! v2, v1, origin
        # ! batch_x, ori_mask = to_dense_batch(batch.x, batch.batch)
        batch_x, ori_mask = to_dense_batch(h_local, batch.batch)
        mask = (~ori_mask).unsqueeze(1).to(dtype=batch.x.dtype) * -1e9
        # * S for node cluster allocation matrix
        vns_emb, S = self.vn_pool_pma(batch_x, None, mask)
        # vns_emb, _ = self.vn_pool_sab(vns_emb, None, None)
        # vns_emb = vns_emb.squeeze(1)  # * [B, k, D]

        # * Multiple VN Mixer
        # * ========================================
        # mixed_vn_emb = self.vn_mixer(vns_emb)  # TODO: change to Transformer

        # * Transformer on VNs
        # * ========================================
        # Multi-head attention.
        # ! BUG vns_emb mask ?
        if self.self_attn is not None:
            mask = None
            if self.global_model_type == "Transformer":
                vns_emb = self._sa_block(vns_emb, None, mask)
            elif self.global_model_type == "Performer":
                vns_emb = self.self_attn(vns_emb, mask=mask)
            elif self.global_model_type == "BigBird":
                vns_emb = self.self_attn(vns_emb, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            vns_emb = self.dropout_attn(vns_emb)
            # vns_emb = h_in1 + vns_emb  # Residual connection.
            # if self.layer_norm:
            #     vns_emb = self.norm1_attn(vns_emb, batch.batch)
            # if self.batch_norm:
            #     vns_emb = self.norm1_attn(vns_emb)
            # Feed Forward block.

            vns_emb = vns_emb + self._ff_block(vns_emb)
            # if self.layer_norm:
            #     vns_emb = self.norm2(vns_emb, batch.batch)
            # if self.batch_norm:
            #     vns_emb = self.norm2(vns_emb)

        # * Add Back
        # * ========================================
        h = reduce(
            einsum(
                rearrange(S, "(b h) c n -> h b c n", h=self.pool_num_heads),
                vns_emb,
                "h b c n, b c d -> h b n d",
            ),
            "h b n d -> b n d",
            "sum",
        )[ori_mask]

        h_out_list.append(h)

        # * Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, "
            f"heads={self.num_heads}"
        )
        return s
