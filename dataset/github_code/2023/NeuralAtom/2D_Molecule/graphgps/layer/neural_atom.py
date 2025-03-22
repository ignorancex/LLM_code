import math
from typing import Optional, Tuple, Type

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear
from torch_geometric.utils import to_dense_batch


class BaseMHA(torch.nn.Module):
    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        Conv: Optional[Type] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) /
                              math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out, A

class Exchanging(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = BaseMHA(in_channels, in_channels, out_channels, num_heads,
                       Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(x, x, graph, mask)

class Porjecting(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_seeds: int,
        Conv: Optional[Type] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = BaseMHA(
            channels, channels, channels, num_heads, Conv=Conv, layer_norm=layer_norm
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(
            self.S.repeat(x.size(0), 1, 1), x, graph, mask
        )  # * S-> Q; x -> K, V
