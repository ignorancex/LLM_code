import torch

from ..builder import HEADS
from .sim_head import SimHead
from eunet.models.utils import FFN, Normalizer
from eunet.datasets.utils.hood_common import NodeType
from eunet.core import multi_apply

from eunet.models.utils.dgl_graph import VERT_ID, MESH_OBJ_ID, DIRECT_FORCE_ID


@HEADS.register_module()
class AccDecoder(SimHead):
    def __init__(self,
                 out_channels=3,
                 in_channels=128*2,
                 dt=1/30,
                 init_cfg=None,
                 eps=1e-7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_acc_steps=None,
                 *args,
                 **kwargs):
        super(AccDecoder, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dt = dt
        self.eps = eps
        self.loss_dim = (0, 3)

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={out_channels} must be a positive integer')

        self.dynamic_proj = FFN([in_channels, in_channels, in_channels, out_channels], final_act=False, act_cfg=act_cfg, add_residual=False)
        self.acc_normalizer = Normalizer(out_channels) if norm_acc_steps is None else Normalizer(out_channels, max_accumulations=norm_acc_steps)

    def init_weights(self):
        super(AccDecoder, self).init_weights()
    
    def evaluate(self, pred, gt_label, **kwargs):
        acc_dict = self.accuracy(pred, gt_label, **kwargs)
        return acc_dict

    def simple_test(self,
            pred, base_graph,
            gt_label=None, test_cfg=None, **kwargs):
        bs = len(pred)
        rst = dict(dynamic=pred)
        acc_rst = dict()
        if gt_label is not None:
            acc_dict = self.evaluate(
                [pred[i][:, :3] for i in range(bs)], 
                [gt_label[i][:, :3] for i in range(bs)],
                term_filter=['dynamic'],
                **kwargs)
            acc_rst.update(acc_dict)
        rst.update(dict(acc=acc_rst))
        return rst

    def pre_predict(self, base_graph, gt_label=None, register_norm=False, state_dim=None, **kwargs):
        bs = len(base_graph)
        pred, base_graph = multi_apply(
            self.predict,
            base_graph, state_dim=state_dim, **kwargs)
        
        if gt_label is not None and register_norm:
            _ = multi_apply(
                self.register_acc_norm,
                base_graph, gt_label)[0]
        
        return pred, base_graph
    
    def register_acc_norm(self, base_graph, gt_label):
        # Not real ground truth, but references
        vertices_nids = torch.nonzero(base_graph.ndata[VERT_ID][:, 0], as_tuple=False)[:, 0]
        state = base_graph.nodes[vertices_nids].data['state']
        gt_pos = gt_label[:, :3]
        prev_pos = state[:, :3]
        prev_prev_pos = state[:, 6:9]
        prev_vel = (prev_pos - prev_prev_pos) / self.dt
        gt_acc = ((gt_pos-prev_pos)/self.dt - prev_vel)/self.dt
        _ = self.acc_normalizer(gt_acc)
        return _,

    def forward_train(self,
            pred, base_graph, potential_prior,
            gt_label, trans, **kwargs):
        
        state_dim = gt_label[0].shape[-1]
        bs = len(pred)
        state, h_state = [], []
        template = []
        mass = []
        gravity = []
        external = []
        for g in base_graph:
            nids = torch.nonzero(g.ndata[VERT_ID][:, 0], as_tuple=False)[:, 0]
            h_nids = torch.nonzero(g.ndata[MESH_OBJ_ID][:, 0], as_tuple=False)[:, 0]
            s = g.nodes[nids].data['state']
            h_s = g.nodes[h_nids].data['state']
            tmlt = g.nodes[nids].data['template']
            template.append(tmlt)
            state.append(s)
            h_state.append(h_s)
            g_mass = g.nodes[nids].data['mass']
            mass.append(g_mass)
            ext = g.nodes[nids].data['external']
            external.append(ext)
            # Gravity
            gravity_nids = torch.nonzero(g.ndata[DIRECT_FORCE_ID][:, 0], as_tuple=False)[:, 0]
            grav = g.nodes[gravity_nids].data['gravity']
            gravity.append(grav)

        losses = dict()
        energy_losses = self.loss(
            [pred[i][:, :3] for i in range(bs)], [gt_label[i][:, :3] for i in range(bs)],
            term_filter=['dynamic'],
            h_state=h_state, state=state, mass=mass, gravity=gravity,
            potential_prior=[pp[:, :1] for pp in potential_prior] if potential_prior is not None else [None]*bs, dt=self.dt, trans=trans,
            frame_dim=state_dim,
            base_graph=base_graph, external=external, **kwargs)
        losses.update(energy_losses)

        return losses, dict(dynamic=pred)

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    def node_pred_vert(self, f_field, dynamic_field):
        def func(nodes):
            vert_emb = nodes.data[f_field]
            feature = vert_emb
            pred_acc = self.dynamic_proj(feature)
            pred_acc = self.acc_normalizer.inverse(pred_acc)
            return {dynamic_field: pred_acc}
        return func
    
    def predict(self, base_graph, state_dim, **kwargs):
        vertices_nids = torch.nonzero(base_graph.ndata[VERT_ID][:, 0], as_tuple=False)[:, 0]
        base_graph.apply_nodes(
            self.node_pred_vert('dyn_vert_feature', 'pred_acc'), vertices_nids)
        
        # Vertices
        pred_acc = base_graph.nodes[vertices_nids].data['pred_acc']
        state = base_graph.nodes[vertices_nids].data['state']
        prev_prev_pos = state[:, state_dim:state_dim+3]
        prev_pos = state[:, :3]
        prev_vel = (prev_pos - prev_prev_pos)/self.dt
        pred_vel = prev_vel+pred_acc*self.dt
        pred_pos = prev_pos + pred_vel*self.dt
        
        vertex_type = base_graph.nodes[vertices_nids].data['vertex_type']
        vert_mask = vertex_type != NodeType.HANDLE
        pred_pos = pred_pos * vert_mask + prev_pos * torch.logical_not(vert_mask)
        pred_vel = (pred_pos - prev_pos) / self.dt

        base_graph.nodes[vertices_nids].data['pred_pos'] = pred_pos
        pred = torch.cat([pred_pos, pred_vel], dim=-1)

        return pred, base_graph

