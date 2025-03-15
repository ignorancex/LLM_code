import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, ModuleList

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from eunet.models.utils import (FFN, Normalizer)
import dgl
import dgl.function as fn
from eunet.models.utils.dgl_graph import VERT_ID, MESH_EDGE, FORCE_EDGE, MESH_OBJ_ID
from eunet.datasets.utils.hood_common import NodeType


class MeshGraphNetEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(MeshGraphNetEncoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg

        self.norms = ModuleList()
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

        # For receiver and sender
        self.mesh_weight = FFN([embed_dims*3, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=False, final_act=False)
        self.world_weight = FFN([embed_dims*3, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=False, final_act=False)
        self.node_weight = FFN([embed_dims*3, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=True, final_act=False)

    def interact_feature(self, mlp_func, norm_func, edge_field, src_field, dst_field, out_field):
        def func(edges):
            sender = edges.src[src_field]
            receiver = edges.dst[dst_field]
            interactions = edges.data[edge_field]
            f = mlp_func(torch.cat([interactions, receiver, sender], dim=-1), residual=interactions)
            f_normed = norm_func(f)
            return {out_field: f_normed}
        return func
    
    def node_feature(self, mlp_func, norm_func, node_field, mesh_field, world_field, out_field):
        def func(nodes):
            node_f = nodes.data[node_field]
            mesh_f = nodes.data[mesh_field]
            world_f = nodes.data[world_field]
            in_msg_emb = torch.cat([mesh_f, world_f], dim=-1)
            f = mlp_func(torch.cat([node_f, in_msg_emb], dim=-1), residual=node_f)
            f_normed = norm_func(f)
            return {out_field: f_normed}
        return func

    def forward(self, g, nids, f_nids, mesh_eids, world_eids, out_vert_field, out_mesh_field, out_world_field, **kwargs):
        inp_edge_feature = g.edata[out_mesh_field]
        g.apply_edges(self.interact_feature(self.mesh_weight, self.norms[0], out_mesh_field, out_vert_field, out_vert_field, out_mesh_field), mesh_eids)
        g.send_and_recv(mesh_eids, fn.copy_e(out_mesh_field, out_mesh_field), fn.sum(out_mesh_field, out_mesh_field))
        if world_eids.shape[0] > 0:
            in_world_feature = g.edata[out_world_field]
            g.apply_edges(self.interact_feature(self.world_weight, self.norms[1], out_world_field, out_vert_field, out_vert_field, out_world_field), world_eids)
            g.send_and_recv(world_eids, fn.copy_e(out_world_field, out_world_field), fn.sum(out_world_field, out_world_field))
        else:
            g.ndata[out_world_field] = torch.zeros_like(g.ndata[out_mesh_field]).to(g.ndata[out_mesh_field])
        g.apply_nodes(self.node_feature(self.node_weight, self.norms[2], out_vert_field, out_mesh_field, out_world_field, out_vert_field), nids)
        g.apply_nodes(self.node_feature(self.node_weight, self.norms[2], out_vert_field, out_mesh_field, out_world_field, out_vert_field), f_nids)

        g.edata[out_mesh_field] += inp_edge_feature
        if world_eids.shape[0] > 0:
            g.edata[out_world_field] += in_world_feature
        return g


class MeshGraphNetEncoder(BaseModule):
    def __init__(self,
                 num_layers,
                 embed_dims,
                 dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(MeshGraphNetEncoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                MeshGraphNetEncoderLayer(embed_dims, dropout, act_cfg, norm_cfg))

    def forward(self, g, nids, f_nids, mesh_eids, world_eids, **kwargs):
        for layer in self.layers:
            g = layer(g, nids, f_nids, mesh_eids, world_eids, **kwargs)
        return g


@BACKBONES.register_module()
class MeshGraphNet(BaseBackbone):
    def __init__(self,
                 state_dim=6,
                 gravity_dim=6,
                 position_dim=3,
                 num_frames=1+1,
                 embed_dims=128,
                 num_encoder_layers=15,
                 dropout=0.0,
                 eps=1e-7,
                 num_fcs=3,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 norm_acc_steps=None,
                 dt=1/30):
        super(MeshGraphNet, self).__init__()
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.eps = eps
        self.gravity_dim = gravity_dim
        self.num_frames = num_frames
        assert num_frames == 2
        self.dt = dt

        # Node
        self.nodetype_embedding = nn.Embedding(NodeType.SIZE, NodeType.SIZE, max_norm=1.)
        node_feature_dim = 3+1+3+NodeType.SIZE+3+3
        self.node_normalizer = Normalizer(node_feature_dim-3) if norm_acc_steps is None else Normalizer(node_feature_dim-3, max_accumulations=norm_acc_steps)
        self.node_encoder = FFN(
            [node_feature_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=False, bias=True)
        self.node_norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # Edge
        mesh_in_dim = (3+1)*2+3
        self.mesh_encoder = FFN(
            [mesh_in_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=False, bias=True)
        self.mesh_normalizer = Normalizer(mesh_in_dim-3) if norm_acc_steps is None else Normalizer(mesh_in_dim-3, max_accumulations=norm_acc_steps)
        world_in_dim = (3+1)*2
        self.world_encoder = FFN(
            [world_in_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=False, bias=True)
        self.world_normalizer = Normalizer(world_in_dim) if norm_acc_steps is None else Normalizer(world_in_dim, max_accumulations=norm_acc_steps)
        self.mesh_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.world_norm = build_norm_layer(norm_cfg, embed_dims)[1]

        self.encoder = MeshGraphNetEncoder(num_encoder_layers, embed_dims, dropout, act_cfg, norm_cfg)

    def embed(self, labels, embedding_layer):
        emb_matrix = embedding_layer.weight.clone().t()
        assert not torch.any(torch.isnan(emb_matrix))
        N = emb_matrix.shape[1]

        if len(labels.shape) == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        labels_onehot = torch.nn.functional.one_hot(labels, N).t().float()
        embedding = emb_matrix @ labels_onehot
        embedding = embedding.t()
        return embedding
    
    def init_edge_features(self,
                           mlp_func, norm_func, normalizer,
                           state_field, template_field,
                           lamemu_field, lamelambda_field, bending_field, out_field):
        def func(edges):
            recv_state = edges.dst[state_field]
            send_state = edges.src[state_field]
            recv_tem = edges.dst[template_field]
            send_tem = edges.src[template_field]
            n_edges = recv_state.shape[0]

            lame_mu = (edges.src[lamemu_field]+edges.dst[lamemu_field]) / 2
            lame_lambda = (edges.src[lamelambda_field]+edges.dst[lamelambda_field]) / 2
            bending = (edges.src[bending_field]+edges.dst[bending_field]) / 2

            delta_state = (recv_state - send_state).reshape(n_edges, self.num_frames, -1, 3)
            delta_pos = delta_state[:, :, 0]
            norm_delta_pos = torch.linalg.norm(delta_pos, dim=-1, keepdim=True)
            in_pos = torch.cat([delta_pos[:, 0], norm_delta_pos[:, 0]], dim=-1).reshape(n_edges, -1)

            delta_tem = recv_tem - send_tem
            norm_delta_tem = torch.linalg.norm(delta_tem, dim=-1, keepdim=True)

            in_raw = normalizer(torch.cat([in_pos, delta_tem, norm_delta_tem], dim=-1))
            in_emb = torch.cat([in_raw, lame_mu, lame_lambda, bending], dim=-1)
            out_emb = norm_func(mlp_func(in_emb))
            
            return {out_field: out_emb}
        return func
    
    def init_world_features(self, mlp_func, norm_func, normalizer, state_field, worldtype_field, out_field):
        def func(edges):
            recv_state = edges.dst[state_field]
            send_state = edges.src[state_field]
            n_edges = recv_state.shape[0]
            recv_garment = edges.dst[worldtype_field] == 0
            send_garment = edges.src[worldtype_field] == 0

            assert torch.sum(send_garment) == 0

            recv_pos = recv_state.reshape(n_edges, self.num_frames, -1, 3)[:, :, 0]
            send_pos = send_state.reshape(n_edges, self.num_frames, -1, 3)[:, :, 0]

            # recv garment, send human: cur-next, cur-cur
            direct1 = torch.stack([recv_pos[:, 0]-send_pos[:, 0], recv_pos[:, 0]-send_pos[:, 1]], dim=1)
            norm_direct1 = torch.linalg.norm(direct1, dim=-1, keepdim=True)
            direct1_emb = torch.cat([direct1, norm_direct1], dim=-1).reshape(n_edges, -1)
            # recv human, send garment: next-cur, cur-cur
            direct2 = torch.stack([recv_pos[:, 0]-send_pos[:, 0], recv_pos[:, 1]-send_pos[:, 0]], dim=1)
            norm_direct2 = torch.linalg.norm(direct2, dim=-1, keepdim=True)
            direct2_emb = torch.cat([direct2, norm_direct2], dim=-1).reshape(n_edges, -1)

            in_emb = recv_garment * direct1_emb + send_garment * direct2_emb
            in_emb = normalizer(in_emb)
            out_emb = norm_func(mlp_func(in_emb))
            
            return {out_field: out_emb}
        return func
    
    def init_node_features(self, state_field, mass_field, normal_field, bending_field, lamemu_field, lamelambda_field, nodetype_field, out_field, external_field=None, gravity_filed=None):
        def func(nodes):
            ## Velocity
            state = nodes.data[state_field]
            assert state.shape[-1] == self.state_dim * self.num_frames
            vel = state[:, :3] - state[:, self.state_dim:self.state_dim+3]
            if self.dt is not None:
                vel = vel / self.dt
            v_mass = nodes.data[mass_field]
            ## Normals
            vertex_normals = nodes.data[normal_field]
            lame_mu = nodes.data[lamemu_field]
            lame_lambda = nodes.data[lamelambda_field]
            bending_coeff = nodes.data[bending_field]
            ## Node Type
            node_type = nodes.data[nodetype_field]
            vertex_type_emb = self.embed(node_type, self.nodetype_embedding)
            # Clean vel
            vel[torch.where(node_type[..., 0] == NodeType.HANDLE)] = 0.0
            in_raw = torch.cat([vel, v_mass, vertex_normals, vertex_type_emb], dim=-1)
            gravity_force = nodes.data[gravity_filed][:, -3:]
            total_force = gravity_force
            if external_field is not None:
                external_force = nodes.data[external_field]
                total_force = external_force+gravity_force
            else:
                total_force = torch.zeros_like(total_force).to(total_force)
            in_raw = torch.cat([in_raw, total_force], dim=-1)
            in_raw = self.node_normalizer(in_raw)
            in_emb = torch.cat([in_raw, lame_mu, lame_lambda, bending_coeff], dim=-1)
            in_feature = self.node_norm(self.node_encoder(in_emb))
            return {out_field: in_feature}
        return func

    def init_features(self, graph,
                      state_field, template_field,
                      nodetype_field, normal_field, mass_field, lame_mu_field, lame_lambda_field, bending_field,
                      out_vert_field,
                      out_mesh_field, out_world_field,
                      eids, f_eids,
                      external_field,
                      gravity_filed='gravity',
                      **kwargs):
        # Parse
        g_nids = torch.nonzero(graph.ndata[VERT_ID][:, 0], as_tuple=False)[:, 0]
        h_nids = torch.nonzero(graph.ndata[MESH_OBJ_ID][:, 0], as_tuple=False)[:, 0]
        ## Garments
        graph.apply_nodes(self.init_node_features(
            state_field, mass_field, normal_field,
            bending_field, lame_mu_field, lame_lambda_field,
            nodetype_field, out_vert_field, external_field=external_field, gravity_filed=gravity_filed), g_nids)
        ## Human nodes
        graph.apply_nodes(self.init_node_features(
            state_field, mass_field, normal_field,
            bending_field, lame_mu_field, lame_lambda_field,
            nodetype_field, out_vert_field, external_field=None, gravity_filed=gravity_filed), h_nids)
        
        # Init edges
        graph.apply_edges(self.init_edge_features(
            self.mesh_encoder, self.mesh_norm, self.mesh_normalizer,
            state_field, template_field,
            lame_mu_field, lame_lambda_field, bending_field, out_mesh_field), eids)
        assert len(f_eids.shape) > 0, f"Got invalid f_eids: {f_eids} with shape: {f_eids.shape}"
        if f_eids.shape[0] > 0:
            graph.apply_edges(self.init_world_features(
                self.world_encoder, self.world_norm, self.world_normalizer,
                state_field, MESH_OBJ_ID, out_world_field), f_eids)
        return graph

    def forward(self, graph, **kwargs):
        external_field = 'external'
        g = dgl.batch(graph)
        nids = torch.nonzero(g.ndata[VERT_ID][:, 0], as_tuple=False)[:, 0]
        f_nids = torch.nonzero(g.ndata[MESH_OBJ_ID][:, 0], as_tuple=False)[:, 0]
        mesh_eids = torch.nonzero(g.edata[MESH_EDGE][:, 0], as_tuple=False)[:, 0]
        if FORCE_EDGE not in g.edata.keys():
            world_eids = torch.empty(0).to(mesh_eids.device)
        else:
            world_eids = torch.nonzero(g.edata[FORCE_EDGE][:, 0], as_tuple=False)[:, 0]
        
        g = self.init_features(g,
                               'state', 'template',
                               'vertex_type', 'normal','mass',
                               'lame_mu', 'lame_lambda', 'bending_coef',
                               'dyn_vert_feature', 'dyn_mesh_feature', 'dyn_world_feature', eids=mesh_eids, f_eids=world_eids, external_field=external_field, **kwargs)
        # Propagate
        g_enc = self.encoder(g, nids, f_nids, mesh_eids, world_eids,
                             out_vert_field='dyn_vert_feature', out_mesh_field='dyn_mesh_feature', out_world_field='dyn_world_feature', **kwargs)
        base_g_list = dgl.unbatch(g_enc)

        return dict(base_graph=base_g_list)