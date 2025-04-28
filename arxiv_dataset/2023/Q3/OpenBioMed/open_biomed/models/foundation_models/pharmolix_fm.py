from typing import Dict, List, Optional
from typing_extensions import Any

import copy
import numpy as np
import logging
from rdkit import Chem, Geometry, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn, knn_graph
from torch_scatter import scatter_mean, scatter_sum
from tqdm import tqdm

from open_biomed.data import Molecule, Pocket, fix_valence, estimate_ligand_atom_num
from open_biomed.models.task_models import PocketMolDockModel, StructureBasedDrugDesignModel
from open_biomed.utils.collator import PygCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MoleculeFeaturizer, PocketFeaturizer, Featurized
from open_biomed.utils.misc import safe_index

# Featurizers
class PharmolixFMMoleculeFeaturizer(MoleculeFeaturizer):
    def __init__(self, pos_norm=1.0, num_node_types=12, num_edge_types=6) -> None:
        super().__init__()
        self.atomic_numbers = [6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34]
        self.mol_bond_types = [
            'empty',
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        self.pos_norm = pos_norm
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        molecule._add_rdmol()
        rdmol = molecule.rdmol
        node_type_list = []
        for atom in rdmol.GetAtoms():
            node_type_list.append(safe_index(self.atomic_numbers, atom.GetAtomicNum()))
        node_type = F.one_hot(torch.LongTensor(node_type_list), num_classes=self.num_node_types).float()
        num_nodes = node_type.shape[0]

        if molecule.conformer is not None:
            pos = torch.tensor(molecule.conformer).float()
        else:
            pos = torch.zeros(num_nodes, 3)
        # Move to center
        pos -= pos.mean(0)
        pos /= self.pos_norm

        # Build halfedge
        if len(rdmol.GetBonds()) <= 0:
            halfedge_index = torch.empty((2, 0), dtype=torch.long)
            halfedge_type = torch.empty(0, dtype=torch.long)
        else:
            halfedge_matrix = torch.zeros([num_nodes, num_nodes], dtype=torch.long)
            for bond in rdmol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = safe_index(self.mol_bond_types, bond.GetBondType())
                halfedge_matrix[i, j] = bond_type
                halfedge_matrix[j, i] = bond_type
            halfedge_index = torch.triu_indices(num_nodes, num_nodes, offset=1)
            halfedge_type = F.one_hot(halfedge_matrix[halfedge_index[0], halfedge_index[1]], num_classes=self.num_edge_types).float()
        
        # Is peptide
        if getattr(molecule, "is_peptide", False):
            is_peptide = torch.ones(num_nodes, dtype=torch.long)
        else:
            is_peptide = torch.zeros(num_nodes, dtype=torch.long)
        
        return Data(**{
            "pos": pos,
            "node_type": node_type,
            "halfedge_type": halfedge_type,
            "halfedge_index": halfedge_index,
            "is_peptide": is_peptide,
        })

    def decode(self, preds: Dict[str, torch.Tensor], pocket_center: Optional[List[float]]) -> Optional[Molecule]:
        pos = preds["pos"] * self.pos_norm
        if pocket_center is not None:
            pos += pocket_center
        num_atoms = pos.shape[0]
        
        for key in preds:
            preds[key] = preds[key].cpu().numpy()

        # Add atoms and coordinates
        rdmol = Chem.RWMol()
        conf = Chem.Conformer()
        for i in range(num_atoms):
            atom = Chem.Atom(self.atomic_numbers[preds["node_type"][i]])
            rdmol.AddAtom(atom)
            atom_pos = Geometry.Point3D(*pos[i].tolist())
            conf.SetAtomPosition(i, atom_pos)
        rdmol.AddConformer(conf)

        # Add bonds
        bond_index = torch.triu_indices(num_atoms, num_atoms, offset=1).numpy()
        for i in range(bond_index.shape[1]):
            st, ed = bond_index[0][i], bond_index[1][i]
            if preds["halfedge_type"][i] > 0:
                rdmol.AddBond(int(st), int(ed), self.mol_bond_types[preds["halfedge_type"][i]])

        # Check validity and fix N valence
        mol = rdmol.GetMol()
        try:
            Chem.SanitizeMol(copy.deepcopy(mol))
            Chem.SanitizeMol(mol)
            return Molecule.from_rdmol(mol)
        except Exception as e:
            mol, fixed = fix_valence(mol)
            if not fixed:
                logging.warn("Failed to generate valid molecule")
                return None
            return Molecule.from_rdmol(mol)

class PharmolixFMPocketFeaturizer(PocketFeaturizer):
    def __init__(self, knn: int=32, pos_norm: float=1.0) -> None:
        super().__init__()
        
        self.knn = knn
        self.pos_norm = pos_norm
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16])    # C, N, O, S
        self.max_num_aa = 20

    def __call__(self, pocket: Pocket) -> Dict[str, Any]:
        elements = torch.LongTensor([atom["atomic_number"] for atom in pocket.atoms])
        elements_one_hot = (elements.view(-1, 1) == self.atomic_numbers.view(1, -1)).long()
        aa_type = torch.LongTensor([atom["aa_type"] for atom in pocket.atoms])
        aa_one_hot = F.one_hot(aa_type, num_classes=self.max_num_aa)
        is_backbone = torch.LongTensor([atom["is_backbone"] for atom in pocket.atoms]).unsqueeze(-1)
        
        x = torch.cat([elements_one_hot, aa_one_hot, is_backbone], dim=-1).float()
        pos = torch.tensor(pocket.conformer, dtype=torch.float32)
        knn_edge_index = knn_graph(pos, k=self.knn, flow='target_to_source')
        pocket_center = pos.mean(dim=0)
        pos -= pocket_center
        pos /= self.pos_norm

        return Data(**{
            "atom_feature": x,
            "knn_edge_index": knn_edge_index,
            "pos": pos,
            "pocket_center": pocket_center.unsqueeze(0),
            "estimated_ligand_num_atoms": torch.tensor(estimate_ligand_atom_num(pocket)).unsqueeze(0),
        })

# Model Layers
class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""
    NONLINEARITIES = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "softplus": nn.Softplus(),
        "elu": nn.ELU(),
        'silu': nn.SiLU()
    }

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(self.NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, type_='exp'):
        super().__init__()
        self.start = start
        self.stop = stop
        if type_ == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start+1), end=np.log(stop+1), steps=num_gaussians)) - 1
        elif type_ == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise NotImplementedError('type_ must be either exp or linear')
        diff = torch.diff(offset)
        diff = torch.cat([diff[:1], diff])
        coeff = -0.5 / (diff**2)
        self.register_buffer('coeff', coeff)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_min(self.start)
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ContextNodeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, gate_dim,
                 context_dim=0, context_edge_dim=0, layernorm_before=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.gate_dim = gate_dim
        self.context_dim = context_dim
        self.context_edge_dim = context_edge_dim
        self.layernorm_before = layernorm_before
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = nn.Linear(hidden_dim, hidden_dim)

        if self.gate_dim > 0:
            self.gate = MLP(edge_dim+(node_dim+gate_dim)*2, hidden_dim, hidden_dim)

        self.centroid_lin = nn.Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.act = nn.ReLU()
        # self.out_transform = Linear(hidden_dim, node_dim)
        self.out_layer = MLP(hidden_dim, node_dim, hidden_dim)
        
        if self.context_dim > 0:
            self.ctx_node_net = MLP(context_dim, hidden_dim, hidden_dim)
            self.ctx_edge_net = MLP(context_edge_dim, hidden_dim, hidden_dim)
            self.ctx_msg_net = nn.Linear(hidden_dim, hidden_dim)
            self.ctx_gate = MLP(context_dim+context_edge_dim+(node_dim+gate_dim), hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, node_extra,
                ctx_x=None, ctx_edge_index=None, ctx_edge_attr=None):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        h_node = self.node_net(x)  # (N, H)

        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H_per_head)
        msg_j = self.msg_net(h_edge + h_node[col] + h_node[row])

        if self.gate_dim > 0:
            gate = self.gate(torch.cat([edge_attr, x[col], node_extra[col], x[row], node_extra[row]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg
        
        # context messages
        if ctx_x is not None:
            row, col = ctx_edge_index
            h_ctx = self.ctx_node_net(ctx_x)
            h_ctx_edge = self.ctx_edge_net(ctx_edge_attr)
            msg_ctx = self.ctx_msg_net(h_ctx_edge * h_ctx[col])
            if self.gate_dim > 0:
                gate = self.ctx_gate(torch.cat([ctx_edge_attr, ctx_x[col], x[row], node_extra[row]], dim=-1))
                msg_ctx = msg_ctx * torch.sigmoid(gate)
            aggred_ctx_msg = scatter_sum(msg_ctx, row, dim=0, dim_size=N)
            out = out + aggred_ctx_msg

        # output. skip connection
        out = self.out_layer(out)
        if not self.layernorm_before:
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out) + x
        return out

class BondFFN(nn.Module):
    def __init__(self, bond_dim, node_dim, inter_dim, gate_dim, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.gate_dim = gate_dim
        self.bond_linear = nn.Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = nn.Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.gate_dim > 0:
            self.gate = MLP(bond_dim+node_dim+gate_dim, out_dim, 32)

    def forward(self, bond_feat_input, node_feat_input, extra):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat + node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.gate_dim > 0:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, extra], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class EdgeBlock(nn.Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, gate_dim=0, layernorm_before=False):
        super().__init__()
        self.gate_dim = gate_dim
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim
        self.layernorm_before = layernorm_before

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, gate_dim=gate_dim)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, gate_dim=gate_dim)

        self.msg_left = nn.Linear(edge_dim, edge_dim)
        self.msg_right = nn.Linear(edge_dim, edge_dim)

        self.node_ffn_left = nn.Linear(node_dim, edge_dim)
        self.node_ffn_right = nn.Linear(node_dim, edge_dim)

        self.self_ffn = nn.Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_layer = MLP(edge_dim, edge_dim, edge_dim)

    def forward(self, h_bond, bond_index, h_node, bond_extra):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_extra)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_extra)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond_update = (
            self.msg_left(msg_bond_left)
            + self.msg_right(msg_bond_right)
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond_update = self.out_layer(h_bond_update)

        # skip connection
        if not self.layernorm_before:
            h_bond = self.layer_norm(h_bond_update + h_bond)
        else:
            h_bond = self.layer_norm(h_bond_update) + h_bond
        return h_bond

class PosUpdate(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, gate_dim, node_dim_right=None):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, node_dim, hidden_dim)
        node_dim_right = node_dim if node_dim_right is None else node_dim_right
        self.right_lin_edge = MLP(node_dim_right, node_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, node_dim*2, node_dim, gate_dim, out_dim=1)
        self.pos_scale_net = nn.Sequential(MLP(node_dim+1+2, 1, hidden_dim), nn.Sigmoid())

    def forward(self, h_node, h_edge, edge_index, relative_vec, distance, node_extra, edge_extra=None, h_node_right=None):
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        h_node_right = h_node if h_node_right is None else h_node_right
        right_feat = self.right_lin_edge(h_node_right[edge_index_right])
        both_extra = node_extra[edge_index_left]
        if edge_extra is not None:
            both_extra = torch.cat([both_extra, edge_extra], dim=-1)
        weight_edge = self.edge_lin(h_edge,
                            torch.cat([left_feat, right_feat], dim=-1),
                            both_extra)
        
        force_edge = weight_edge * relative_vec / (distance.unsqueeze(-1) + 1e-6) / (distance.unsqueeze(-1) + 5.) * 5
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])
        delta_pos = delta_pos * self.pos_scale_net(torch.cat([h_node, node_extra,
                                        torch.norm(delta_pos, dim=-1, keepdim=True)], dim=-1))
        return delta_pos

class ContextNodeEdgeNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim,
                 num_blocks, dist_cfg, gate_dim=0,
                 context_dim=0, context_cfg=None,
                 node_only=False, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.dist_cfg = dist_cfg
        self.gate_dim = gate_dim
        self.node_only = node_only
        self.kwargs = kwargs
        self.downsample_context = kwargs.get('downsample_context', False)
        self.layernorm_before = kwargs.get("layernorm_before", False)

        self.distance_expansion = GaussianSmearing(**dist_cfg)
        num_gaussians = dist_cfg['num_gaussians']
        input_edge_dim = num_gaussians + (0 if node_only else edge_dim)
            
        # for context
        self.context_cfg = context_cfg
        if context_cfg is not None:
            context_edge_dim = context_cfg['edge_dim']
            self.knn = context_cfg['knn']
            self.dist_exp_ctx = GaussianSmearing(**context_cfg['dist_cfg'])
            input_context_edge_dim = context_cfg['dist_cfg']['num_gaussians']
            assert context_dim > 0, 'context_dim should be larger than 0 if context_cfg is not None'
            assert not node_only, 'not support node_only with context'
        else:
            context_edge_dim = 0
        
        # node network
        self.edge_embs = nn.ModuleList()
        self.node_blocks_with_edge = nn.ModuleList()
        if not node_only:
            self.edge_blocks = nn.ModuleList()
            self.pos_blocks = nn.ModuleList()
            if self.context_cfg is not None:
                self.ctx_edge_embs = nn.ModuleList()
                self.ctx_pos_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # edge emb
            self.edge_embs.append(nn.Linear(input_edge_dim, edge_dim))
            # node update
            self.node_blocks_with_edge.append(ContextNodeBlock(
                node_dim, edge_dim, hidden_dim, gate_dim,
                context_dim, context_edge_dim, layernorm_before=self.layernorm_before
            ))
            if node_only:
                continue
            # edge update
            self.edge_blocks.append(EdgeBlock(
                edge_dim=edge_dim, node_dim=node_dim, gate_dim=gate_dim, layernorm_before=self.layernorm_before
            ))
            # pos update
            self.pos_blocks.append(PosUpdate(
                node_dim, edge_dim, hidden_dim=edge_dim, gate_dim=gate_dim*2,
            ))
            if self.context_cfg is not None:
                self.ctx_edge_embs.append(nn.Linear(input_context_edge_dim, context_edge_dim))
                self.ctx_pos_blocks.append(PosUpdate(
                    node_dim, context_edge_dim, hidden_dim=edge_dim, gate_dim=gate_dim,
                    node_dim_right=context_dim,
                ))
                
    def forward(self, h_node, pos_node, h_edge, edge_index,
                node_extra, edge_extra, batch_node=None,
                h_ctx=None, pos_ctx=None, batch_ctx=None):
        """
        graph node/edge features
            h_node: (n_node, node_dim)
            pos_node: (n_node, 3)
            h_edge: (n_edge, edge_dim)
            edge_index: (2, n_edge)
            node_extra: (n_node, node_extra_dim)
            edge_extra: (n_edge, edge_extra_dim)
            batch_node: (n_node, )
        context node features
            h_ctx: (n_ctx, ctx_dim)
            pos_ctx: (n_ctx, 3)
            batch_ctx: (n_ctx, )
        Output:
            h_node: (n_node, node_dim)
            h_edge: (n_edge, edge_dim)
            pos_node: (n_node, 3)
        """

        for i in range(self.num_blocks):
            # # remake edge fetures (distance have been changed in each iteration)
            if (i==0) or (not self.node_only):
                h_dist, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            if not self.node_only:
                h_edge = torch.cat([h_edge, h_dist], dim=-1)
            else:
                h_edge = h_dist
            h_edge = self.edge_embs[i](h_edge)
            
            # # edge with context
            if h_ctx is not None:
                h_ctx_edge, vec_ctx, dist_ctx, ctx_knn_edge_index = self._build_context_edges_dist(
                    pos_node, pos_ctx, batch_node, batch_ctx)
                h_ctx_edge = self.ctx_edge_embs[i](h_ctx_edge)
            else:
                ctx_knn_edge_index = None
                h_ctx_edge = None

            # # node feature updates
            h_node = self.node_blocks_with_edge[i](h_node, edge_index, h_edge, node_extra,
                                        h_ctx, ctx_knn_edge_index, h_ctx_edge)
            if self.node_only:
                continue
            
            # # edge feature updates
            h_edge = self.edge_blocks[i](h_edge, edge_index, h_node, edge_extra)

            # # pos updates
            pos_node = pos_node + self.pos_blocks[i](h_node, h_edge, edge_index, relative_vec, distance, node_extra, edge_extra)
            if h_ctx is not None:
                pos_node = pos_node + self.ctx_pos_blocks[i](
                    h_node, h_ctx_edge, ctx_knn_edge_index, vec_ctx, dist_ctx, node_extra,
                    edge_extra=None, h_node_right=h_ctx)

        if self.node_only:
            return h_node
        else:
            return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.distance_expansion(distance)
        return h_dist, relative_vec, distance
    
    def _build_context_edges_dist(self, pos, pos_ctx, batch_node, batch_ctx):
        # build knn edge index
        if self.knn < 100:
            if self.downsample_context:
                pos_ctx_noised = pos_ctx + torch.randn_like(pos_ctx) * 5  # works like masked position information
            else:
                pos_ctx_noised = pos_ctx
            ctx_knn_edge_index = knn(y=pos, x=pos_ctx_noised, k=self.knn,
                                    batch_x=batch_ctx, batch_y=batch_node)
        else: # fully connected x-yf
            device = pos.device
            ctx_knn_edge_index = []
            cum_node = 0
            cum_ctx = 0
            for i_batch in range(batch_ctx.max()+1):
                num_ctx = (batch_ctx==i_batch).sum()
                num_node = (batch_node==i_batch).sum()
                ctx_knn_edge_index_this = torch.stack(
                    torch.meshgrid(
                        torch.arange(num_node, device=device) + cum_node,
                        torch.arange(num_ctx, device=device) + cum_ctx,
                    )).view(2, -1)
                cum_node += num_node
                cum_ctx += num_ctx
                ctx_knn_edge_index.append(ctx_knn_edge_index_this)
            ctx_knn_edge_index = torch.cat(ctx_knn_edge_index, dim=-1)

        relative_vec = pos[ctx_knn_edge_index[0]] - pos_ctx[ctx_knn_edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.dist_exp_ctx(distance)
        return h_dist, relative_vec, distance, ctx_knn_edge_index

class PharmolixFM(PocketMolDockModel, StructureBasedDrugDesignModel):
    def __init__(self, model_cfg: Config) -> None:
        super(PharmolixFM, self).__init__(model_cfg)
        self.config = model_cfg
        self.num_node_types = model_cfg.num_node_types
        self.num_edge_types = model_cfg.num_edge_types
        
        # # pocket encoder
        pocket_dim = model_cfg.pocket_dim
        self.pocket_embedder = nn.Linear(model_cfg.pocket_in_dim, pocket_dim)
        self.pocket_encoder = ContextNodeEdgeNet(pocket_dim, node_only=True, **model_cfg.pocket.todict())

        # # mol embedding
        self.addition_node_features = getattr(model_cfg, 'addition_node_features', [])
        node_dim = model_cfg.node_dim
        edge_dim = model_cfg.edge_dim
        node_emb_dim = node_dim - len(self.addition_node_features)
        edge_emb_dim = edge_dim 
        self.nodetype_embedder = nn.Linear(self.num_node_types + 2, node_emb_dim) # t_pos_in and t_node_in 
        self.edgetype_embedder = nn.Linear(self.num_edge_types + 2, edge_emb_dim) # t_halfedge_in and fixed_halfdist

        # # bfn backbone
        self.denoiser = ContextNodeEdgeNet(node_dim, edge_dim,
                            context_dim=pocket_dim, **model_cfg.denoiser.todict())

        # # decoder for discrete variables
        self.node_decoder = MLP(node_dim, self.num_node_types, node_dim)
        self.edge_decoder = MLP(edge_dim, self.num_edge_types, edge_dim)

        # additional output
        self.add_output = getattr(model_cfg, 'add_output', [])
        if 'confidence' in self.add_output:  # condidence
            self.node_cfd = MLP(node_dim, 1, node_dim // 2)
            self.pos_cfd = MLP(node_dim, 1, node_dim // 2)
            self.edge_cfd = MLP(edge_dim, 1, edge_dim // 2)

        self.featurizers = {
            "molecule": PharmolixFMMoleculeFeaturizer(model_cfg.pos_norm),
            "pocket": PharmolixFMPocketFeaturizer(model_cfg.pocket_knn, model_cfg.pos_norm),
        }
        self.collators = {
            "molecule": PygCollator(follow_batch=["pos", "node_type", "halfedge_type"]),
            "pocket": PygCollator(follow_batch=["pos"])
        }

        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def continuous_var_bayesian_update(self, t: torch.Tensor, x: torch.Tensor, fixed_pos: torch.Tensor=None, orig_x: torch.Tensor=None) -> torch.Tensor:
        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
        gamma = (1 - torch.pow(self.config.sigma1, 2 * t)).unsqueeze(1)  # [B]
        mu = gamma * x + torch.sqrt((gamma + 0.01) * (1 - gamma)) * torch.randn_like(x)
        if fixed_pos is not None:
            mu[torch.where(fixed_pos)] = orig_x[torch.where(fixed_pos)]
        return mu, gamma

    def discrete_var_bayesian_update(self, t: torch.Tensor, x: torch.Tensor, K: int, fixed_pos: torch.Tensor=None, orig_x: torch.Tensor=None) -> torch.Tensor:
        # Eq.(182): β(t) = t**2 β(1)
        beta = (self.config.beta1 * (t**2)).unsqueeze(1)  # (B,)

        # Eq.(185): p_F(θ|x;t) = E_{N(y | β(t)(Ke_x−1), β(t)KI)} δ (θ − softmax(y))
        # can be sampled by first drawing y ~ N(y | β(t)(Ke_x−1), β(t)KI)
        # then setting θ = softmax(y)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        if fixed_pos is not None:
            theta[torch.where(fixed_pos)] = orig_x[torch.where(fixed_pos)]
        return theta

    def create_dummy_molecule(self, pocket: Featurized[Pocket]) -> Featurized[Molecule]:
        num_atoms = pocket["estimated_ligand_num_atoms"].cpu()
        num_halfedge = (num_atoms ** 2 - num_atoms) // 2
        batch_size = num_atoms.shape[0]
        return Data(**{
            "pos": torch.randn(num_atoms.sum().item(), 3) * 0.01,
            "node_type": torch.ones(num_atoms.sum().item(), self.num_node_types) / self.num_node_types,
            "halfedge_type": torch.ones(num_halfedge.sum().item(), self.num_edge_types) / self.num_edge_types,
            "is_peptide": torch.zeros(num_atoms.sum().item(), dtype=torch.long),
            "halfedge_index": torch.cat([torch.triu_indices(num, num, offset=1) for num in num_atoms], dim=1),
            "pos_batch": torch.repeat_interleave(torch.arange(batch_size), num_atoms),
            "node_type_batch": torch.repeat_interleave(torch.arange(batch_size), num_atoms),
            "halfedge_type_batch": torch.repeat_interleave(torch.arange(batch_size), num_halfedge),
        }).to(pocket["atom_feature"].device)

    def model_forward(self, 
        molecule: Featurized[Molecule],
        pocket: Optional[Featurized[Pocket]],
    ) -> Dict[str, torch.Tensor]:
        pos_in = molecule['pos']
        
        if len(molecule['node_type'].shape) <= 1:
            molecule['node_type'] = F.one_hot(molecule['node_type'], self.num_node_types).float()
            molecule['halfedge_type'] = F.one_hot(molecule['halfedge_type'], self.num_edge_types).float()
        if 't_node_type' not in molecule:
            molecule['t_node_type'] = torch.ones(molecule['node_type'].shape[0], dtype=torch.float, device=molecule['node_type'].device)
            molecule['t_halfedge_type'] = torch.ones(molecule['halfedge_type'].shape[0], dtype=torch.float, device=molecule['halfedge_type'].device)

        h_node_in = molecule['node_type']
        h_halfedge_in = molecule['halfedge_type']

        # add t indicator as extra features
        node_extra = torch.stack([molecule['t_node_type'], molecule['t_pos']], dim=1).to(pos_in.dtype)
        halfedge_extra = torch.stack([molecule['t_halfedge_type'], molecule['fixed_halfdist']], dim=1).to(pos_in.dtype)
        h_node_in = torch.cat([h_node_in, node_extra], dim=-1)
        h_halfedge_in = torch.cat([h_halfedge_in, halfedge_extra], dim=-1)

        # from 1/K \in [0,1] to 2/K-1 \in [-1,1]
        h_node_in = self.nodetype_embedder(2 * h_node_in - 1)
        h_halfedge_in = self.edgetype_embedder(2 * h_halfedge_in - 1)

        # break symmetry
        n_halfedges = h_halfedge_in.shape[0]
        halfedge_index = molecule['halfedge_index']
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
        edge_extra = torch.cat([halfedge_extra, halfedge_extra], dim=0)

        # additonal node features
        if 'is_peptide' in self.addition_node_features:
            is_peptide = molecule['is_peptide'].unsqueeze(-1).to(pos_in.dtype)
            h_node_in = torch.cat([h_node_in, is_peptide], dim=-1)

        # # encode pocket
        h_pocket = self.pocket_embedder(pocket['atom_feature'])
        h_pocket = self.pocket_encoder(
            h_node=h_pocket,
            pos_node=pocket['pos'],
            edge_index=pocket['knn_edge_index'],
            h_edge=None,
            node_extra=None,
            edge_extra=None,
        )

        # # 2 interdependency modeling
        h_node, pred_pos, h_edge = self.denoiser(
            h_node=h_node_in,
            pos_node=pos_in, 
            h_edge=h_edge_in, 
            edge_index=edge_index,
            node_extra=node_extra,
            edge_extra=edge_extra,
            batch_node=molecule['node_type_batch'],
            # pocket
            h_ctx=h_pocket,
            pos_ctx=pocket['pos'],
            batch_ctx=pocket['pos_batch'],
        )

        # # 3 predict the variables
        # for discrete, we take softmax
        node_logits = self.node_decoder(h_node)
        pred_node = F.softmax(node_logits, dim=-1)
        halfedge_logits = self.edge_decoder(h_edge[:n_halfedges] + h_edge[n_halfedges:])
        pred_halfedge = F.softmax(halfedge_logits, dim=-1)

        additional_outputs = {}
        if 'confidence' in self.add_output:
            pred_node_cfd = self.node_cfd(h_node)
            pred_pos_cfd = self.pos_cfd(h_node)  # use the node hidden
            pred_edge_cfd = self.edge_cfd(h_edge[:n_halfedges]+h_edge[n_halfedges:])  # NOTE why not divide by 2?
            additional_outputs = {'confidence_node_type': pred_node_cfd, 'confidence_pos': pred_pos_cfd, 'confidence_halfedge_type': pred_edge_cfd}

        return {
            'pos': pred_pos,
            'node_type': pred_node,
            'halfedge_type': pred_halfedge,
            **additional_outputs,
        }

    @torch.no_grad()
    def sample(self,
        molecule: Featurized[Molecule],
        pocket: Optional[Featurized[Pocket]]=None,
    ) -> List[Molecule]:
        # Initialization
        device = molecule['pos'].device
        
        molecule_in = {
            "pos": torch.randn_like(molecule['pos']) * 0.01,
            "node_type": torch.ones_like(molecule['node_type']) / self.config.num_node_types,
            "halfedge_type": torch.ones_like(molecule['halfedge_type']) / self.config.num_edge_types,
        }
        in_traj, out_traj, cfd_traj = {}, {}, {}
        for key in molecule_in:
            fixed = molecule[f"fixed_{key}"]
            if fixed is not None:
                molecule_in[key][fixed] = molecule[key][fixed]
            molecule[key] = molecule_in[key]
            in_traj[key] = []
            out_traj[key] = []
            cfd_traj[key] = []

        # BFN update
        for step in tqdm(range(1, self.config.num_sample_steps + 1), desc="Sampling"):
            t = torch.ones(1, dtype=torch.float, device=device) * (step - 1) / self.config.num_sample_steps
            t_in = {
                "pos": t.repeat(molecule["pos"].shape[0]),
                "node_type": t.repeat(molecule["pos"].shape[0]),
                "halfedge_type": t.repeat(molecule["halfedge_type"].shape[0]),
            }
            for key in t_in:
                fixed = molecule[f"fixed_{key}"]
                t_in[key][torch.where(fixed)] = 1
                molecule[f"t_{key}"] = t_in[key]
            outputs_step = self.model_forward(molecule, pocket)
            for key in t_in:
                in_traj[key].append(copy.deepcopy(molecule[key]))
                out_traj[key].append(copy.deepcopy(outputs_step[key]))
                cfd_traj[key].append(copy.deepcopy(outputs_step[f"confidence_{key}"]))

            # Destination prediction
            if step == self.config.num_sample_steps:
                for key in t_in:
                    molecule[key] = outputs_step[key]
                continue

            molecule["pos"], _ = self.continuous_var_bayesian_update(
                t_in["pos"], outputs_step["pos"], 
                fixed_pos=molecule["fixed_pos"], orig_x=molecule["pos"]
            )
            molecule["node_type"] = self.discrete_var_bayesian_update(
                t_in["node_type"], outputs_step["node_type"], self.num_node_types, 
                fixed_pos=molecule["fixed_node_type"], orig_x=molecule["node_type"]
            )
            molecule["halfedge_type"] = self.discrete_var_bayesian_update(
                t_in["halfedge_type"], outputs_step["halfedge_type"], self.num_edge_types, 
                fixed_pos=molecule["fixed_halfedge_type"], orig_x=molecule["halfedge_type"]
            )

        # Split and reconstruct molecule
        num_mols = molecule["node_type_batch"].max() + 1
        in_traj_split, out_traj_split, cfd_traj_split = [], [], []
        out_molecules = []
        for i in tqdm(range(num_mols), desc="Post processing molecules..."):
            in_traj_split.append({})
            out_traj_split.append({})
            cfd_traj_split.append({})
            cur_molecule = {}
            for key in molecule_in:
                idx = torch.where(molecule[f"{key}_batch"] == i)
                in_traj_split[i][key] = [val[idx] for val in in_traj[key]]
                out_traj_split[i][key] = [val[idx] for val in out_traj[key]]
                cfd_traj_split[i][key] = [val[idx] for val in cfd_traj[key]]
                cur_molecule[key] = out_traj_split[i][key][-1]
            cur_molecule["node_type"] = torch.argmax(cur_molecule["node_type"], dim=-1)
            cur_molecule["halfedge_type"] = torch.argmax(cur_molecule["halfedge_type"], dim=-1)
            out_molecules.append(self.featurizers["molecule"].decode(cur_molecule, pocket["pocket_center"][i]))
            # print(out_molecules[-1].conformer)
        return out_molecules

    # TODO: implement training of PharMolixFM
    def forward_pocket_molecule_docking(self, pocket: Featurized[Pocket], label: Featurized[Molecule]) -> Dict[str, torch.Tensor]:
        pass

    def forward_structure_based_drug_design(self, pocket: List[Pocket], label: List[Molecule]) -> Dict[str, torch.Tensor]:
        pass

    def predict_pocket_molecule_docking(self,
        molecule: Featurized[Molecule],
        pocket: Featurized[Pocket],
    ) -> List[Molecule]:
        molecule["fixed_pos"] = torch.zeros(molecule["pos"].shape[0], dtype=torch.bool, device=molecule["pos"].device)
        molecule["fixed_node_type"] = torch.ones(molecule["node_type"].shape[0], dtype=torch.bool, device=molecule["node_type"].device)
        molecule["fixed_halfedge_type"] = torch.ones(molecule["halfedge_type"].shape[0], dtype=torch.bool, device=molecule["halfedge_type"].device)
        molecule["fixed_halfdist"] = torch.zeros(molecule["halfedge_type"].shape[0], device=molecule["halfedge_type"].device)
        return self.sample(molecule, pocket)

    def predict_structure_based_drug_design(self, 
        pocket: Featurized[Pocket]
    ) -> List[Molecule]:
        molecule = self.create_dummy_molecule(pocket)
        molecule["fixed_pos"] = torch.zeros(molecule["pos"].shape[0], dtype=torch.bool, device=molecule["pos"].device)
        molecule["fixed_node_type"] = torch.zeros(molecule["node_type"].shape[0], dtype=torch.bool, device=molecule["node_type"].device)
        molecule["fixed_halfedge_type"] = torch.zeros(molecule["halfedge_type"].shape[0], dtype=torch.bool, device=molecule["halfedge_type"].device)
        molecule["fixed_halfdist"] = torch.zeros(molecule["halfedge_type"].shape[0], device=molecule["halfedge_type"].device)
        return self.sample(molecule, pocket)