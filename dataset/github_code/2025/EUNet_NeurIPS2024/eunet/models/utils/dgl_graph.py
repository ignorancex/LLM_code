import torch

from mmcv.ops import QueryAndGroup

import dgl
from eunet.core import multi_apply
from eunet.utils import vertex_normal_batched_simple

from ..builder import PREPROCESSOR

RECEIVER_ID = 'receiver'
SENDER_ID = 'sender'

MESH_OBJ_ID = 'vert_mesh'
DIRECT_FORCE_ID = 'vert_invisable'

VERT_ID = 'vertices'
MESH_EDGE = 'vert2vert'

FORCE_ID = 'forces'
FORCE_EDGE = 'force_e'


@PREPROCESSOR.register_module()
class DynamicDGLProcessor(object):
    def __init__(self, eps=1e-7, **kwargs) -> None:
        self.eps = eps
        self.graph_builder = BuildDGLGraph(**kwargs)
    
    def _preprocess(self, 
                    state, mass, templates, edges,
                    vertex_type, vertex_level,
                    lame_mu, lame_lambda, bending_coef,
                    h_state, faces, h_faces, gravity,
                    # For HOOD construction
                    bistride0_mask_sparse=None,
                    bistride1_mask_sparse=None,
                    bistride2_mask_sparse=None,
                    vert_mask=None,
                    external=None,
                    human_vert_mask=None):
        # Calculate vertex_normals
        g_vert_normals = vertex_normal_batched_simple(state[None, :, :3], faces[None, :])[0]
        # Human inputs normals is next state state
        h_vert_normals = vertex_normal_batched_simple(h_state[None, :, :3], h_faces[None, :])[0]
        num_g = state.shape[0]
        if external is not None:
            node_external = external
        else:
            node_external = torch.zeros_like(g_vert_normals).to(g_vert_normals)
        node_wise_dict = {
            'mass': mass,
            'vertex_type': vertex_type[:num_g],
            'vertex_level':vertex_level[:num_g],
            'lame_mu': lame_mu.repeat(num_g, 1),
            'lame_lambda': lame_lambda.repeat(num_g, 1),
            'bending_coef': bending_coef.repeat(num_g, 1),
            'template': templates,
            'state': state,
            'normal': g_vert_normals,
            'external': node_external,
            'gravity': torch.tile(gravity, (num_g, 1)).to(gravity),
            VERT_ID: torch.ones((num_g, 1)).to(state.device)}
        
        force_scalar = -1.0
        force_ndata_list = [
            {'state': h_state,
             # Default inputs <<<
             'mass': force_scalar * torch.ones((h_state.shape[0], 1)).to(h_state),
             'lame_mu': force_scalar * torch.ones((h_state.shape[0], 1)).to(h_state),
             'lame_lambda': force_scalar * torch.ones((h_state.shape[0], 1)).to(h_state),
             'bending_coef': force_scalar * torch.ones((h_state.shape[0], 1)).to(h_state),
             # >>>
             'normal': h_vert_normals,
             'vertex_type': vertex_type[num_g:],
             'vertex_level':vertex_level[num_g:],
             'human_vert_mask': torch.ones((h_state.shape[0], 1)).to(h_state) if human_vert_mask is None else human_vert_mask,
             FORCE_ID: torch.ones((h_state.shape[0], 1)).to(h_state), MESH_OBJ_ID: torch.ones((h_state.shape[0], 1)).to(h_state)},
            # Not used for interaction for now
            {'gravity': gravity, FORCE_ID: torch.ones((gravity.shape[0], 1)).to(gravity), DIRECT_FORCE_ID: torch.ones((gravity.shape[0], 1)).to(gravity)},
        ]
        force_topo_list = [None, None]
        force_edata_list = [None, None]
        
        g = self.graph_builder.build_graph(
            node_wise_dict, edges,
            force_topo_list, force_ndata_list, force_edata_list,
            bistride0_mask_sparse=bistride0_mask_sparse, bistride1_mask_sparse=bistride1_mask_sparse, bistride2_mask_sparse=bistride2_mask_sparse)
        return g,
    
    def batch_preprocess(self,
                         state, mass, templates, edges,
                         vertex_type, vertex_level,
                         lame_mu, lame_lambda, bending_coef,
                         faces, h_faces, h_state, gravity,
                         bistride0_mask_sparse=None, bistride1_mask_sparse=None, bistride2_mask_sparse=None,
                         trans=None, gt_label=None, vert_mask=None, external=None, human_vert_mask=None, **kwargs):
        bs = len(state)
        g_list = multi_apply(
            self._preprocess,
            state, mass, templates, edges,
            vertex_type, vertex_level,
            lame_mu, lame_lambda, bending_coef,
            h_state, faces, h_faces, gravity,
            bistride0_mask_sparse if bistride0_mask_sparse is not None else [None]*bs,
            bistride1_mask_sparse if bistride1_mask_sparse is not None else [None]*bs,
            bistride2_mask_sparse if bistride2_mask_sparse is not None else [None]*bs,
            vert_mask if vert_mask is not None else [None]*bs,
            external if external is not None else [None]*bs,
            human_vert_mask if human_vert_mask is not None else [None]*bs)[0]
        g_dict = dict(graph=g_list)
        meta_dict = dict(gt_label=gt_label, trans=trans, faces=faces, h_faces=h_faces, vert_mask=vert_mask)
        meta_dict.update(kwargs)
        return g_dict, meta_dict

@PREPROCESSOR.register_module()
class DynamicContactProcessor(object):
    def __init__(self, radius, group_cfg, **kwargs) -> None:
        super(DynamicContactProcessor, self).__init__(**kwargs)
        self.radius = radius
        self.grouper = QueryAndGroup(**group_cfg)
        group_cfg['sample_num'] = 1
        self.nearest_grouper = QueryAndGroup(**group_cfg)

    def _dynamic_edges(self, receiver_pos, sender_pos, non_intersect_relations=None, remove_self=True):
        group_xyz_diff, group_idx = self.grouper(
            sender_pos.contiguous(),
            receiver_pos.contiguous())
        group_xyz_diff = group_xyz_diff.permute(0, 2, 3, 1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < self.radius
        group_xyz_l2_mask = group_xyz_l2_mask.squeeze(0).squeeze(-1)
        valid_group_idx = group_idx.squeeze(0)
        valid_group_xyz_l2_mask = group_xyz_l2_mask
        valid_idx = torch.where(valid_group_xyz_l2_mask == True)
        valid_neighbor = valid_group_idx[valid_idx].to(torch.int64)
        relation_pair = torch.unique(torch.stack([valid_idx[0], valid_neighbor], dim=-1), dim=0)
        # Remove self-loop
        if remove_self:
            relation_non_loop = torch.where(relation_pair[:, 0] != relation_pair[:, 1])
            relation_pair = relation_pair[relation_non_loop]
        # Clear unique pairs
        if non_intersect_relations is not None:
            candidates = torch.cat([relation_pair.transpose(-1, -2), non_intersect_relations, non_intersect_relations], dim=-1).transpose(-1, -2)
            r_uniques, r_counts = torch.unique(candidates, dim=0, return_counts=True)
            r_pairs = r_uniques[r_counts == 1]
            relation_pair = r_pairs
        return relation_pair

    def _g_preprocess(self, graph_item, state_dim):
        g_nids = torch.nonzero(graph_item.ndata[VERT_ID][:, 0], as_tuple=False)[:, 0]
        o_nids = torch.nonzero(graph_item.ndata[MESH_OBJ_ID][:, 0], as_tuple=False)[:, 0]
        mesh_eids = torch.nonzero(graph_item.edata[MESH_EDGE][:, 0], as_tuple=False)[:, 0]

        g_pos = graph_item.nodes[g_nids].data['state'][:, :3].unsqueeze(0)
        assert graph_item.nodes[o_nids].data['state'].shape[-1] == state_dim * 2
        o_pos = graph_item.nodes[o_nids].data['state'][:, state_dim:state_dim+3].unsqueeze(0)
        non_intersect_relations = torch.stack(graph_item.edges(form='uv'), dim=0)[:, mesh_eids]
        g2g_edges = self._dynamic_edges(g_pos, g_pos, non_intersect_relations=non_intersect_relations)

        selected_human = o_pos
        g2o_edges = self._dynamic_edges(g_pos, selected_human, non_intersect_relations=None, remove_self=False)
        g_offset = torch.min(g_nids)
        o_offset = torch.min(o_nids)
        g2g_edges += g_offset
        g2o_edges[:, 0] += g_offset
        g2o_edges[:, 1] += o_offset
        if g2o_edges.shape[0] <= 0:
            # Add dummy edges here
            graph_item.add_edges(
                o_offset, g_offset,
                data={FORCE_EDGE: torch.zeros((1, 1)).to(g_pos)}
            )
        else:
            graph_item.add_edges(
                g2o_edges[:, 1], g2o_edges[:, 0],
                data={FORCE_EDGE: torch.ones((g2o_edges.shape[0], 1)).to(g_pos)})
        
        return graph_item,

    def graph_preprocess(self, graph_dict, state_dim, **kwargs):
        '''
            The keys need to align with DGLProcessor
        '''
        graph_list = graph_dict['graph']
        graph_list = multi_apply(
            self._g_preprocess,
            graph_list,
            state_dim=state_dim)[0]

        graph_dict['graph'] = graph_list
        return graph_dict

class BuildDGLGraph:
    def __init__(self, receiver_id=None, sender_id=None) -> None:
        self.receiver_id = VERT_ID
        self.sender_id = VERT_ID
        if receiver_id is not None:
            self.receiver_id = receiver_id
        if sender_id is not None:
            self.sender_id = sender_id

    def build_graph(self,
            node_wise_dict, edges,
            force_topo_list, force_node_data_list, force_edge_data_list,
            bistride0_mask_sparse=None, bistride1_mask_sparse=None, bistride2_mask_sparse=None):
        # Static graph
        base_g = self.base_graph(node_wise_dict, edges)
        if bistride0_mask_sparse is not None:
            base_g = self.bistride_graph(base_g, bistride0_mask_sparse, HOOD_HIE_V0, HOOD_HIE_E0)
        if bistride1_mask_sparse is not None:
            base_g = self.bistride_graph(base_g, bistride1_mask_sparse, HOOD_HIE_V1, HOOD_HIE_E1)
        if bistride2_mask_sparse is not None:
            base_g = self.bistride_graph(base_g, bistride2_mask_sparse, HOOD_HIE_V2, HOOD_HIE_E2)
        
        # Graph for nn modeling
        nids = torch.nonzero(base_g.ndata[self.receiver_id][:, 0], as_tuple=False)[:, 0]
        node_offset = torch.min(nids)
        ## Between human, gravity
        for f_topo, f_ndata, f_edata in zip(force_topo_list, force_node_data_list, force_edge_data_list):
            base_g = self.force_graph(
                base_g, f_topo,
                node_offset=node_offset, force_node_data_dict=f_ndata, force_edge_data_dict=f_edata)
        return base_g

    def bistride_graph(self, base_g, g_mask, vert_id, edge_id):
        mesh_rel = g_mask['indices']
        if 'values' not in g_mask.keys():
            m_values = torch.ones((mesh_rel.shape[-1], 1)).to(mesh_rel.device)
        base_g.add_edges(
            mesh_rel[1], mesh_rel[0],
            data={
                edge_id: m_values})
        # Mask nodes
        v_ids = torch.unique(mesh_rel[0])
        base_g.nodes[v_ids].data[vert_id] = torch.ones((v_ids.shape[0], 1)).to(v_ids.device)
        return base_g

    def base_graph(self, node_wise_dict, edges):
        m_value = torch.ones((edges.shape[-1], 1)).to(edges.device)
        ## Build graph: src, dst
        g = dgl.graph((edges[0], edges[1]))
        g.edata[MESH_EDGE] = m_value

        assert VERT_ID in node_wise_dict.keys()
        for key, val in node_wise_dict.items():
            g.ndata[key] = val

        return g

    def force_graph(self, g, force_topo=None, node_offset=0, force_node_data_dict=None, force_edge_data_dict=None):
        cur_graph_nodes = g.num_nodes()
        if force_topo is None:
            assert force_node_data_dict is not None
            force_edge = None
            force_num = force_node_data_dict[FORCE_ID].shape[0]
        else:
            force_edge = force_topo['indices'].clone()
            _, force_num = force_topo['size']
            force_num = force_num.item()

        if force_node_data_dict is not None:
            assert FORCE_ID in force_node_data_dict.keys()
        else:
            force_node_data_dict = {FORCE_ID: torch.ones((force_num, 1)).to(force_edge)}
        if force_edge_data_dict is not None:
            assert FORCE_EDGE in force_edge_data_dict.keys()
        elif force_edge is not None:
            force_edge_data_dict = {FORCE_EDGE: torch.ones((force_edge[0].shape[0], 1)).to(force_edge)}
        
        g.add_nodes(
            force_num,
            data=force_node_data_dict)
        if force_edge is not None:
            force_edge[0] += node_offset
            force_edge[1] += cur_graph_nodes
            g.add_edges(
                force_edge[1], force_edge[0],
                data=force_edge_data_dict)

        return g