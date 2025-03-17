import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tformer import TFormer, MultiHeadAttention
from pformer import PFormer


class TemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=1024):
        super(TemporalEncoding, self).__init__()
        self.max_len = max_len
        position = torch.arange(0., self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(self.max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid)
        scaled_cosine = torch.cos(position * div_term) / math.sqrt(n_hid)
        if n_hid % 2 == 1:
            scaled_cosine = scaled_cosine[:, :-1]
        emb.weight.data[:, 1::2] = scaled_cosine
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t.int())).unsqueeze(1)

    def reset_parameters(self):
        self.lin.reset_parameters()


class TGTOD(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, time_len, num_parts,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add', use_cformer=True, station=True):
        super(TGTOD, self).__init__()
        self.te = TemporalEncoding(hidden_channels, time_len)
        self.pformer = PFormer(in_channels, hidden_channels, hidden_channels,
                               trans_num_layers=trans_num_layers, trans_num_heads=trans_num_heads,
                               trans_dropout=trans_dropout, trans_use_bn=trans_use_bn,
                               trans_use_residual=trans_use_residual, trans_use_weight=trans_use_weight,
                               trans_use_act=trans_use_act, gnn_num_layers=gnn_num_layers,
                               gnn_dropout=gnn_dropout, gnn_use_weight=gnn_use_weight, gnn_use_init=gnn_use_init,
                               gnn_use_bn=gnn_use_bn, gnn_use_residual=gnn_use_residual, gnn_use_act=gnn_use_act,
                               use_graph=use_graph, graph_weight=graph_weight, aggregate=aggregate)

        self.tformer = TFormer(trans_num_heads, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.fuse_lin = nn.Linear(2 * hidden_channels, hidden_channels)
        self.cformer = MultiHeadAttention(trans_num_heads, hidden_channels, trans_dropout)
        self.part_feat = None
        self.num_parts = num_parts
        self.hidden_channels = hidden_channels
        self.use_cformer = use_cformer
        self.station = station

    def forward(self, x, edge_index, i):
        if self.station:
            x = [x] * len(edge_index)
        else:
            x = x.permute(1, 0, 2).contiguous()
        h = torch.stack([self.pformer(xt, ei) for xt, ei in zip(x, edge_index)])
        if self.use_cformer:
            if self.part_feat is None:
                self.part_feat = torch.randn(h.shape[0], self.num_parts, self.hidden_channels, device=x[0].device)

            self.part_feat[:, i, :] = torch.mean(h, dim=-2).detach()
            part_emb, _ = self.cformer(self.part_feat,
                                       self.part_feat,
                                       self.part_feat)
            idx = torch.full((x[0].shape[0],), i)
            p = part_emb[:, idx, :]
            z = torch.cat([h, p], dim=-1)
            h = F.relu(self.fuse_lin(z)) + h

        self.te(h, torch.arange(h.shape[0]).to(h.device))
        h = h.permute(1, 0, 2)
        h = self.tformer(h, None)
        if self.station:
            h = torch.mean(h, dim=1)
        return self.lin(h)

    def reset_parameters(self):
        self.tformer.reset_parameters()
        self.pformer.reset_parameters()
        self.cformer.reset_parameters()
        self.lin.reset_parameters()
        self.te.reset_parameters()
        self.fuse_lin.reset_parameters()
