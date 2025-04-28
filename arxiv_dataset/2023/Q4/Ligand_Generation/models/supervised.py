import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from gvp.models import ThreeD_Protein_Model
from torch_geometric.graphgym.models import AtomEncoder, BondEncoder
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class LigandGraphEncoder(nn.Module):
    def __init__(self, emb_dim = 32, hidden_dim = 128, num_layer = 3):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layer = num_layer
        for i in range(self.num_layer):
            if i == 0:
                node_fdim = emb_dim
            else:
                node_fdim = hidden_dim
            self.convs.append(GATConv(
                in_channels = node_fdim, out_channels = hidden_dim, 
                heads = 4, concat = False, dropout = 0.1, edge_dim = emb_dim
            ))
            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

    def forward(self, lig_graph):
        lig_graph = self.atom_encoder(lig_graph)
        lig_graph = self.bond_encoder(lig_graph)
        x = lig_graph.x
        for i in range(self.num_layer):
            x = self.convs[i](x, lig_graph.edge_index, lig_graph.edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
        return global_mean_pool(x, lig_graph.batch)

class BindingAffinityPredictor(nn.Module):
    def __init__(self, hidden_dim = 128, attention_type = "performer", use_mp = False):
        super().__init__()
        self.protein_model = ThreeD_Protein_Model(node_in_dim = (6,3), node_h_dim = (hidden_dim, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                      num_layers = 3, drop_rate=0.1, attention_type = attention_type)

        self.ligand_model = LigandGraphEncoder(emb_dim = 32, hidden_dim = hidden_dim, num_layer = 3)
 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1280, hidden_dim * 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        if use_mp:
            self.ligand_mp = nn.Sequential(
                nn.Linear(2048, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(), 
                nn.Dropout(0.1), 
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        in_dim = hidden_dim * 2 + 1280 if not use_mp else hidden_dim * 3 + 1280
        self.prediction_head = nn.Sequential(
            nn.Linear(in_dim, 1024), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, protein_graph, seq_emb, lig_graph, lig_mp = None):
        x_prot = self.protein_model((protein_graph.node_s, protein_graph.node_v), 
                                   protein_graph.edge_index, (protein_graph.edge_s, protein_graph.edge_v), seq = None, batch = protein_graph.batch)
        x_lig = self.ligand_model(lig_graph) 
        if lig_mp is not None:
            x_mp = self.ligand_mp(lig_mp)
            x = torch.cat([x_prot, seq_emb, x_lig, x_mp], dim = -1)
        else:
            x = torch.cat([x_prot, seq_emb, x_lig], dim = -1)
        return self.prediction_head(x)