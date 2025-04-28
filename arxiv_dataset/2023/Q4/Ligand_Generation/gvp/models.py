import numpy as np
import torch
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean
from performer_pytorch import Performer
from torch_geometric.utils import to_dense_batch
from linear_attention_transformer import LinearAttentionTransformerLM, LinformerSettings


class ThreeD_Protein_Model(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.5, attention_type = "performer"):
        
        super().__init__()
         
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), vector_gate=True))
            
        self.attention_type = attention_type
        if attention_type == "performer":
            self.transformer = Performer(
                            dim = ns,
                            depth = 2,
                            heads = 4,
                            dim_head = ns // 4, 
                            causal = False
                        )
        else:
            layer = nn.TransformerEncoderLayer(ns, 4, ns * 2, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 2)
 
        self.skip_connection = nn.Sequential(nn.Linear(ns * 2, ns), nn.ReLU(), nn.Linear(ns, ns))
         
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        ''' 
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V) 
        x, mask = to_dense_batch(out, batch) 
        x_o = self.transformer(x)
        x = torch.cat([x, x_o], dim = -1)
        x = self.skip_connection(x)
        geo_rep = x.mean(dim = 1)
        if seq is not None:
            z = torch.cat([geo_rep, seq], dim = -1)
            return z
        return geo_rep