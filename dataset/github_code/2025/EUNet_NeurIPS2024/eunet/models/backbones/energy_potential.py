import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, ModuleList

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from eunet.models.utils import (FFN, Normalizer)

from eunet.utils import face_normals_batched, vertex_normal_batched_simple
from eunet.datasets.utils.hood_common import gather


class GNNEncoderLayer(BaseModule):
    """Implements one encoder layer in transformer.
    Args:
        embed_dims (int): The feature dimension. Same as `FFN`.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
        order (tuple[str]): The order for encoder layer. Valid examples are
            ('selfattn', 'norm', 'ffn', 'norm') and ('norm', 'selfattn',
            'norm', 'ffn'). Default ('selfattn', 'norm', 'ffn', 'norm').
        act_cfg (dict): The activation config for FFNs. Default ReLU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
    """

    def __init__(self,
                 embed_dims,
                 dropout=0.0,
                 act_cfg=dict(type='SiLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 num_fcs=2,
                 **kwargs):
        super(GNNEncoderLayer, self).__init__()
        self.embed_dims = embed_dims

        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = pre_norm
        self.num_fcs = num_fcs

        edge_fuse_in_dim = embed_dims
        self.emb_encoder = FFN(
            [edge_fuse_in_dim] + [embed_dims for _ in range(num_fcs)],
            final_act=True, bias=True, act_cfg=act_cfg, add_residual=True)
        
        norm_emb = edge_fuse_in_dim if pre_norm else embed_dims
        self.edge_norm = build_norm_layer(norm_cfg, norm_emb)[1]

    def forward(self, in_emb, **kwargs):
        # Gather Edge Info from Verts
        inplace_emb = in_emb
        if self.pre_norm:
            in_emb = self.edge_norm(in_emb)
        in_emb = self.emb_encoder(in_emb, residual=inplace_emb)
        if not self.pre_norm:
            in_emb = self.edge_norm(in_emb)

        return in_emb


class ForceEncoder(BaseModule):
    def __init__(self,
                 num_layers,
                 embed_dims,
                 pre_norm=False,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(ForceEncoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.pre_norm = pre_norm

        if pre_norm:
            self.emb_norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = GNNEncoderLayer(embed_dims, norm_cfg=norm_cfg, pre_norm=pre_norm, **kwargs)
            self.layers.append(layer)

    def forward(self, in_emb, **kwargs):
        # Need follow the exact order to apply the different mask
        for i, layer in enumerate(self.layers):
            in_emb = layer(in_emb, **kwargs)
        if self.pre_norm:
            in_emb = self.emb_norm(in_emb)
        return in_emb


@BACKBONES.register_module()
class EnergyPotential(BaseBackbone):
    """Implements the simulation transformer.
    """

    def __init__(self,
                 attr_dim=5,
                 state_dim=3,
                 position_dim=3,
                 num_force_layers=4,
                 init_cfg=None,
                 embed_dims=128,
                 pre_norm=False,
                 eps=1e-7,
                 act_cfg=dict(type='SiLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_fcs=3,
                 norm_acc_steps=100000,
                 dissipate_sigma=0.5,
                 dt=1/30,
                 *args,
                 **kwargs):
        super(EnergyPotential, self).__init__(init_cfg=init_cfg)
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.eps = eps
        self.num_fcs = num_fcs
        self.num_force_layers = num_force_layers

        self.dissipate_sigma = dissipate_sigma
        self.dt = dt
        
        # l2
        edge_info_dim = 1
        ## cos, sin, cos, sin
        extra_theta_dim = 4
        edge_info_dim += extra_theta_dim
        self.edge_pos_encoder = FFN(
            [edge_info_dim+attr_dim] + [embed_dims for i in range(self.num_fcs)],
            final_act=True, bias=True, act_cfg=act_cfg)
        edge_info_norm_dim = edge_info_dim
        edge_info_norm_dim -= (extra_theta_dim)
        self.edge_pos_normalizer = Normalizer(size=edge_info_norm_dim, max_accumulations=norm_acc_steps)

        self.edge_info_norm = nn.Identity()
        if norm_cfg is not None and not pre_norm:
            self.edge_info_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        
        self.force_encoder = ForceEncoder(
            num_force_layers, embed_dims, num_fcs=num_fcs,
            pre_norm=pre_norm, act_cfg=act_cfg, norm_cfg=norm_cfg,
            **kwargs)

        # For dissipate
        if self.dissipate_sigma is not None:
            # Per face encoder
            ## velocity(1), attr_dim
            dissipate_edge_dim = 1+attr_dim
            self.dissipate_edge_encoder = FFN(
                [dissipate_edge_dim] + [embed_dims for i in range(self.num_fcs)],
                final_act=True, bias=False, act_cfg=act_cfg)
            ## Only for velocity
            self.dissipate_edge_normalizer = Normalizer(size=1, max_accumulations=norm_acc_steps)

            self.dissipate_edge_norm = nn.Identity()
            if norm_cfg is not None and not pre_norm:
                self.dissipate_edge_norm = build_norm_layer(norm_cfg, embed_dims)[1]
                
            self.dissipate_force_encoder = ForceEncoder(
                num_force_layers, embed_dims, num_fcs=num_fcs,
                pre_norm=pre_norm, act_cfg=act_cfg, norm_cfg=norm_cfg, hidden_bias=True,
                **kwargs)

    def _cos_sin_l1theta_withdir(self, vec1, vec2, dir_vec):
        norm_1 = torch.linalg.norm(vec1, dim=-1, keepdim=True)
        norm_2 = torch.linalg.norm(vec2, dim=-1, keepdim=True)
        dot_p = torch.sum(vec1*vec2, dim=-1, keepdim=True)
        cos = dot_p / (norm_1*norm_2+self.eps)
        cross_p = torch.cross(vec1, vec2)
        sin_n = cross_p / (norm_1*norm_2+self.eps)
        sin = torch.sum(sin_n * dir_vec, dim=-1, keepdim=True)
        return cos, sin
    
    def _get_rel_rotation_edge_vn_theta(self, vert_normal, states, f_connectivity_edges):
        vn = gather(vert_normal, f_connectivity_edges, 0, 1, 1)
        src_normal, dst_normal = torch.unbind(vn, dim=-2)

        v = gather(states, f_connectivity_edges, 0, 1, 1)
        v0, v1 = torch.unbind(v, dim=-2)
        e = v1 - v0
        l = torch.norm(e, dim=-1, keepdim=True)
        e_norm = e / l

        y_axis = e_norm
        x_axis = torch.cross(y_axis, src_normal)
        x_axis = x_axis / (torch.linalg.norm(x_axis, dim=-1, keepdim=True)+self.eps)
        z_axis = torch.cross(x_axis, y_axis)
        z_axis = z_axis / (torch.linalg.norm(z_axis, dim=-1, keepdim=True)+self.eps)

        # n_edge, stack(3), 3
        base_axis = torch.stack([x_axis, y_axis, z_axis], dim=1)
        # n_edge, 3
        local_dst_normal = torch.bmm(base_axis, dst_normal.unsqueeze(-1)).squeeze(-1)
        local_src_normal = torch.bmm(base_axis, src_normal.unsqueeze(-1)).squeeze(-1)

        device = local_dst_normal.device
        dst_local_zx = torch.cat([local_dst_normal[:, 2:3], torch.zeros_like(local_dst_normal[:, 0:1]).to(device), local_dst_normal[:, 0:1]], dim=-1)
        src_local_zx = torch.cat([local_src_normal[:, 2:3], torch.zeros_like(local_src_normal[:, 0:1]).to(device), local_src_normal[:, 0:1]], dim=-1)
        y_dir = torch.zeros((1, 3)).to(dst_local_zx)
        y_dir[0,1] = 1.0
        cos_alongedge, sin_alongedge = self._cos_sin_l1theta_withdir(src_local_zx, dst_local_zx, y_dir)
        rotY = torch.eye(3).to(local_dst_normal).unsqueeze(0).repeat(local_dst_normal.shape[0], 1, 1)
        rotY[:, 0, 0:1] = cos_alongedge
        rotY[:, 0, 2:3] = sin_alongedge
        rotY[:, 2, 0:1] = -sin_alongedge
        rotY[:, 2, 2:3] = cos_alongedge

        rotated_local_dst_normal = torch.bmm(rotY, local_dst_normal.unsqueeze(-1)).squeeze(-1)
        x_dir = torch.zeros((1, 3)).to(local_dst_normal)
        x_dir[0,0] = 1.0
        cos_plane, sin_plane = self._cos_sin_l1theta_withdir(local_src_normal, rotated_local_dst_normal, x_dir)

        return cos_alongedge, sin_alongedge, cos_plane, sin_plane
    
    def _get_edge_length(self, state, f_connectivity_edges):
        v = gather(state, f_connectivity_edges, 0, 1, 1)
        v0, v1 = torch.unbind(v, dim=-2)
        l2 = torch.linalg.norm(v1-v0, dim=-1, keepdim=True)
        return l2
    
    def _get_edge_vel(self, state, f_connectivity_edges):
        v = gather(state, f_connectivity_edges, 0, 1, 1)
        v0, v1 = torch.unbind(v, dim=-2)
        return (v0+v1)/2
    
    def _get_edge_face_area(self, face_area, f_connectivity):
        n = gather(face_area, f_connectivity, 0, 1, 1)
        n0, n1 = torch.unbind(n, dim=-2)
        return (n0+n1)/2

    def forward(self, states, templates, faces, f_connectivity, f_connectivity_edges, attr, prev_states=None, register_norm=True):
        '''
            states: n_verts, 3
            templates
            faces: n_faces, 3
            attr: 1, 5
            f_connectivity: [n_edges, 2(face_idx)], neighbor face index
            f_connectivity_edges: [n_edges, 2(vert_idx)], neighbor face common verts
        '''
        attr = attr[:, :self.attr_dim]
        n_edges = f_connectivity.shape[0]

        # Face normals
        _, face_area = face_normals_batched(states.unsqueeze(0), faces.unsqueeze(0), with_face_area=True)
        face_area = face_area[0]

        # Bending info
        template_v_normal = vertex_normal_batched_simple(templates.unsqueeze(0), faces.unsqueeze(0))[0]
        vertex_normal = vertex_normal_batched_simple(states.unsqueeze(0), faces.unsqueeze(0))[0]
        tem_cos_ae, tem_sin_ae, tem_cos_plane, tem_sin_plane = self._get_rel_rotation_edge_vn_theta(template_v_normal, templates, f_connectivity_edges)
        cur_cos_ae, cur_sin_ae, cur_cos_plane, cur_sin_plane = self._get_rel_rotation_edge_vn_theta(vertex_normal, states, f_connectivity_edges)
        # cos(a-b) = cosa cosb + sina sinb
        delta_cos_ae = cur_cos_ae*tem_cos_ae + cur_sin_ae*tem_sin_ae
        delta_cos_plane = cur_cos_plane*tem_cos_plane + cur_sin_plane*tem_sin_plane
        # sin(a-b) = sina cosb - cosa sinb
        delta_sin_ae = cur_sin_ae*tem_cos_ae - cur_cos_ae*tem_sin_ae
        delta_sin_plane = cur_sin_plane*tem_cos_plane - cur_cos_plane*tem_sin_plane
        # No need further normalize
        normal_in_feature = torch.cat([delta_cos_ae, delta_sin_ae, delta_cos_plane, delta_sin_plane], dim=-1)

        # Length info
        template_l2 = self._get_edge_length(templates, f_connectivity_edges)
        cur_l2 = self._get_edge_length(states, f_connectivity_edges)
        delta_l2 = cur_l2 - template_l2
        pos_info_vec = delta_l2

        # Encode the edge info
        normed_pos_l2 = self.edge_pos_normalizer(pos_info_vec, omit_overwrite=not register_norm)
        rel_pos_emb_i = self.edge_pos_encoder(torch.cat([normed_pos_l2, attr.repeat(n_edges, 1), normal_in_feature], dim=-1))
        
        edge_feature = rel_pos_emb_i
        edge_emb = edge_feature
        out_emb = self.force_encoder(self.edge_info_norm(edge_emb))

        # For decoder usage
        edge_face_area = self._get_edge_face_area(face_area, f_connectivity)
        if self.dissipate_sigma is not None:
            # Calculate dissipation
            # Use face area for the edges and the mean vel of the edge
            assert prev_states is not None
            vert_vel = (states[:, :3] - prev_states[:, :3]) / self.dt
            edge_vel = self._get_edge_vel(vert_vel, f_connectivity_edges)
            ## n_edges, 1
            evel_descriptor = torch.linalg.norm(edge_vel, dim=-1, keepdim=True)
            in_evel_descriptor_transformed = evel_descriptor
            normed_evel = self.dissipate_edge_normalizer(in_evel_descriptor_transformed, omit_overwrite=not register_norm)
            ## Edge face area
            vel_emb = torch.cat([normed_evel, attr.repeat(n_edges, 1)], dim=-1)
            dissipate_hidden_emb = self.dissipate_edge_norm(self.dissipate_edge_encoder(vel_emb))
            dissipate_out_emb = self.dissipate_force_encoder(dissipate_hidden_emb)
            out_emb = torch.cat([out_emb, dissipate_out_emb], dim=-1)

        return dict(energy_emb=out_emb, face_area=edge_face_area)

    def train(self, mode=True):
        super().train(mode)