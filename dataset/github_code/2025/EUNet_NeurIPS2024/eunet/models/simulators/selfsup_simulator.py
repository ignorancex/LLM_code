import torch
from torch.nn.modules.batchnorm import _BatchNorm
from collections import defaultdict

from .. import builder
from ..builder import SIMULATORS
from .base import BaseSimulator
from eunet.core import add_prefix, multi_apply
from eunet.datasets.utils import to_numpy_detach
from eunet.models.utils.hood_collision import CollisionPreprocessor
from eunet.datasets.utils.hood_common import NodeType

import numpy as np


@SIMULATORS.register_module()
class SelfsupDynamic(BaseSimulator):
    def __init__(self,
                 backbone,
                 decode_head,
                 selfsup_potential_cfg,
                 dynamic_cfg=None,
                 processor_cfg=dict(type='DynamicDGLProcessor'),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 norm_eval=False,
                 collision_pushup=2e-3,):
        super(SelfsupDynamic, self).__init__(init_cfg=init_cfg)
        self.norm_eval = norm_eval

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and simulator set pretrained weight'
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = builder.build_backbone(backbone)
        self.decode_head = builder.build_head(decode_head)

        if isinstance(processor_cfg, dict):
            processor_cfg = [processor_cfg]
        generator_cfg = processor_cfg[0]
        self.graph_generator = builder.build_preprocessor(generator_cfg)
        self.preprocessor = []
        for i in range(1, len(processor_cfg)):
            p_cfg = processor_cfg[i]
            self.preprocessor.append(builder.build_preprocessor(p_cfg))
        # For dyanmic collision detection
        self.dynamic_detector = None
        if dynamic_cfg is not None:
            self.dynamic_detector = builder.build_preprocessor(dynamic_cfg)

        # This is for preprocess/augment train input
        self.train_cfg = train_cfg
        # This is for preprocess/augment test input
        self.test_cfg = test_cfg

        self.potential = None
        if selfsup_potential_cfg is not None:
            self.potential = builder.build_simulator(selfsup_potential_cfg)
            self.potential.init_weights()

        self.collision_solver = None
        if dynamic_cfg is not None:
            self.collision_solver = CollisionPreprocessor(push_eps=collision_pushup)
        
    def _pack_dynamic(self, pred_list, cur_dynamic):
        cur_state = []
        prev_state = []
        for pred, cur_dyn in zip(pred_list, cur_dynamic):
            pred_dim = pred.shape[-1]
            history_rst = cur_dyn[:, :-pred_dim]
            assert history_rst.shape[-1] == pred_dim
            cur_state.append(pred)
            prev_state.append(history_rst)
        return cur_state, prev_state
    
    def _rollout_steps(self, num_epoch, num_iter, cfg):
        step_max_interval = cfg.get('step_max_interval', None)
        max_steps = cfg.get('step', None)
        by_epoch = cfg.get('by_epoch', None)
        assert by_epoch is not None
        assert max_steps is not None and max_steps > 0, "Should be at least 1"
        assert step_max_interval is not None
        if by_epoch:
            counter = num_epoch
        else:
            counter = num_iter
        # Set current predictions of frames
        if step_max_interval == 0:
            rollout_size = max_steps
        else:
            rollout_size = 1 + (counter // step_max_interval)
            rollout_size = min(rollout_size, max_steps)
        return rollout_size, max_steps
    
    def _label_offset(self):
        # 0, 0, 1, 2, 3 instead of 0,1,2,3,4
        return 1
    
    def _process_history(self, inputs, gt_label=None, is_training=False, rollout_idx=0, random_ts=False, **kwargs):
        # During taining in self-supervised manner, gt_label is not real gt but estimated by LBS
        assert gt_label is not None
        state_dim = gt_label['vertices'][0].shape[-1]
        if rollout_idx == 0:
            ## vel(t) is NO USE
            h_state_list = [h_s for h_s in inputs['dynamic']['h_state']]
            for bs_i in range(len(h_state_list)):
                cur_h = h_state_list[bs_i][:, state_dim:state_dim*2]
                next_h = torch.cat([cur_h[:, :3], torch.zeros_like(cur_h[:, 3:]).to(cur_h)], dim=-1)
                h_state_list[bs_i][:, :state_dim] = next_h
            inputs['dynamic']['h_state'] = h_state_list
            cur_states = [torch.cat([bs_dynamic[:, :3], torch.zeros_like(bs_dynamic[:, 3:state_dim]).to(bs_dynamic)], dim=-1) for bs_dynamic in inputs['dynamic']['state']]
            gt_label['vertices'] = cur_states

            if np.random.rand() > 0.5 or not random_ts:
                for bs_i in range(len(h_state_list)):
                    h_state_list[bs_i][:, state_dim+3:2*state_dim] *= 0.0
                inputs['dynamic']['h_state'] = h_state_list
                in_state_dim = inputs['dynamic']['state'][0].shape[-1]
                num_frames = in_state_dim // state_dim
                assert in_state_dim % state_dim == 0
                init_states = [c_s.repeat((1, num_frames)) for c_s in cur_states]
                inputs['dynamic']['state'] = init_states
        elif rollout_idx == 1:
            h_state_list = [h_s for h_s in inputs['dynamic']['h_state']]
            for bs_i in range(len(h_state_list)):
                h_state_list[bs_i][:, state_dim+3:2*state_dim] *= 0
            inputs['dynamic']['h_state'] = h_state_list
            cur_states = [torch.cat([bs_dynamic[:, :3], torch.zeros_like(bs_dynamic[:, 3:state_dim]).to(bs_dynamic), bs_dynamic[:, state_dim:]], dim=-1) for bs_dynamic in inputs['dynamic']['state']]
            inputs['dynamic']['state'] = cur_states

        return inputs, gt_label

    def _pre_maskout_pinverts_rollout(self, input_state, gt_label, vert_mask):
        n_verts, state_dim = gt_label.shape
        in_state = input_state.reshape(n_verts, -1, state_dim)
        cur_state = gt_label
        his_state = in_state[:, :-1].reshape(n_verts, -1)
        pinverts_state = torch.cat([cur_state, his_state], dim=-1)
        rst_state = input_state * vert_mask + pinverts_state * (1-vert_mask)
        return rst_state,

    def _register_dynamic_forces(self, input_graph, input_meta, state_dim, **kwargs):
        if self.dynamic_detector is not None:
            input_graph = self.dynamic_detector.graph_preprocess(input_graph, **input_meta, state_dim=state_dim)
        return input_graph
    
    def _preprocess(self, inputs, gt_label=None, is_training=False, rollout_idx=0, random_ts=False, **kwargs):
        inputs, gt_label = self._process_history(inputs, gt_label=gt_label, is_training=is_training, rollout_idx=rollout_idx, random_ts=random_ts, **kwargs)
        if rollout_idx == 0:
            assert gt_label is not None
            state_dim = gt_label['vertices'][0].shape[-1]
            cur_states = inputs['dynamic']['state']
            h_state = inputs['dynamic']['h_state']
            h_faces = inputs['static']['h_faces']
            faces = inputs['static']['faces']
            cur_states = self.collision_solver.solve(cur_states, h_state, h_faces, state_dim, faces=faces)
            inputs['dynamic']['state'] = cur_states
        # Replace pinned verts given the node type
        nodetype = inputs['static']['vertex_type']
        pinned_mask = [nt != NodeType.HANDLE for nt in nodetype]
        vert_mask = [pm[:g_state.shape[0]].float() for pm, g_state in zip(pinned_mask, inputs['static']['templates'])]
        inputs['static']['vert_mask'] = vert_mask # Overwrite
        in_state = inputs['dynamic'].pop('state', None)
        assert in_state is not None
        future_state = gt_label['vertices']
        rst_state = multi_apply(
            self._pre_maskout_pinverts_rollout,
            in_state, future_state, vert_mask)[0]
        inputs['dynamic']['state'] = rst_state

        graph_dict, meta_dict = self.graph_generator.batch_preprocess(**inputs['dynamic'], **inputs['static'], gt_label=gt_label['vertices'] if gt_label is not None else None)
        state_dim = gt_label['vertices'][0].shape[-1]
        for i in range(len(self.preprocessor)):
            processor = self.preprocessor[i]
            graph_dict = processor.graph_preprocess(graph_dict, is_training=is_training, state_dim=state_dim)
        
        graph_dict = self._register_dynamic_forces(graph_dict, meta_dict, state_dim=state_dim)
        return graph_dict, meta_dict
    
    def _postprocess(self, inputs, pred, gt_label):
        return pred
    
    def rollout_history_input(self, bs, frame_state_dim, pred_dict_i, cur_dynamic, next_dynamic):
        pred_state = []
        for b_i in range(bs):
            history_rst = cur_dynamic['state'][b_i][:, :-frame_state_dim]
            step_rst = pred_dict_i['dynamic'][b_i]
            assert step_rst.shape[-1] == frame_state_dim
            history_rst = history_rst.detach()
            step_rst = step_rst.detach()
            step_rst = torch.cat([step_rst, history_rst], dim=-1)
            pred_state.append(step_rst)
        # Update inputs
        rst_dynamic = next_dynamic
        rst_dynamic['state'] = pred_state
        return rst_dynamic
    
    def extract_feat(self, input_graph, input_meta, **kwargs):
        """Extract features from inputs."""
        x = self.backbone(**input_graph, **input_meta, **kwargs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def _decode_forward_train(self, x, input_graph, input_meta, raw_inputs, raw_gt_label, register_norm=True, state_dim=None, **kwargs):
        losses = dict()
        pred, base_graph = self.decode_head.pre_predict(
            **x, **input_graph, **input_meta,
            train_cfg=self.train_cfg, register_norm=register_norm, state_dim=state_dim)

        # Calcualte the potential
        # Preprocess potentials
        orig_state = raw_inputs['dynamic']['state']
        wrapped_cur_dynamic, wrapped_prev_dynamic = self._pack_dynamic(pred, raw_inputs['dynamic']['state'])
        raw_inputs['dynamic']['state'] = wrapped_cur_dynamic
        raw_inputs['prev_dynamic'] = dict(state=wrapped_prev_dynamic)
        potential_dict = self.potential.inference(raw_inputs, out_energy=True, out_dev=False, **kwargs)
        # Recover back the gt
        _ = raw_inputs.pop('prev_dynamic', None)
        raw_inputs['dynamic']['state'] = orig_state
        potential_list = potential_dict['pred']

        loss_decode, pred_dict = self.decode_head.forward_train(
            pred=pred, base_graph=base_graph, potential_prior=potential_list,
            train_cfg=self.train_cfg, **input_meta)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, pred_dict
    
    def _encode_decode_train(self, inputs, gt_label, register_norm=True, state_dim=None, **kwargs):
        input_graph, input_meta = self._preprocess(inputs, gt_label, **kwargs)
        
        losses = dict()
        # Forward
        features = self.extract_feat(input_graph, input_meta, **kwargs)
        loss_decode, pred_dict = self._decode_forward_train(features, input_graph, input_meta, inputs, gt_label, register_norm=register_norm, state_dim=state_dim) # Only difference: add inputs and gt_label
        losses.update(loss_decode)

        return losses, pred_dict
    
    def forward_train(self, inputs, gt_label, num_epoch=0, num_iter=0, **kwargs):
        rollout_size, max_steps = self._rollout_steps(num_epoch, num_iter, self.train_cfg)
        bs = len(gt_label[0]['vertices'])
        frame_state_dim = gt_label[0]['vertices'][0].shape[1]
        assert frame_state_dim == self.backbone.state_dim
        in_dynamics = inputs['dynamic']
        in_statics = inputs['static']
        assert in_statics.get('num_iter', None) == None
        assert len(in_dynamics) == max_steps
        in_statics['num_iter'] = num_iter

        # Rollout steps
        random_ts = (rollout_size == 1)
        cur_step = 0
        label_step = max(0, cur_step-self._label_offset())
        cur_dynamic = in_dynamics[label_step]
        cur_label = gt_label[label_step]
        # Auto regressive
        losses = defaultdict(list)
        acc_dict = defaultdict(list)
        for g_step in range(rollout_size):
            if label_step == 0:
                cloned_label = dict()
                for key, val in cur_label.items():
                    assert isinstance(val, list)
                    cloned_label[key] = [v.clone() for v in val]
                cloned_dynamic = dict()
                for key, val in cur_dynamic.items():
                    assert isinstance(val, list)
                    cloned_dynamic[key] = [v.clone() for v in val]
            else:
                cloned_label = cur_label
                cloned_dynamic = cur_dynamic
            losses_i, pred_dict_i = self._encode_decode_train(
                dict(dynamic=cloned_dynamic, static=in_statics),
                cloned_label,
                is_training=True, rollout_idx=g_step, random_ts=random_ts, register_norm=True, state_dim=frame_state_dim, **kwargs)
            pred_dict_i = self._postprocess(dict(dynamic=cloned_dynamic, static=in_statics), pred_dict_i, cloned_label)

            # Merge loss
            for key in losses_i.keys():
                if key.startswith('decode.loss'):
                    # Sum loss
                    losses[key].append(losses_i[key])
                else:
                    acc_key = key
                    if g_step > 0:
                        acc_key = f"{key}_step{g_step}"
                    if losses_i[key] > 0:
                        # Meaningful only when larger than 0
                        acc_dict[acc_key].append(losses_i[key])
            
            cur_step += 1
            label_step = max(0, cur_step-self._label_offset())
            if cur_step >= rollout_size:
                break
            next_dynamic = in_dynamics[label_step]
            cur_dynamic = self.rollout_history_input(
                bs, frame_state_dim, pred_dict_i, cloned_dynamic, next_dynamic)
            cur_label = gt_label[label_step]
            
        # reduce losses
        losses_rst = dict()
        for key, val in losses.items():
            losses_rst[key] = torch.sum(torch.stack(val))
        for key, val in acc_dict.items():
            losses_rst[key] = torch.mean(torch.stack(val))

        return losses_rst
    
    def _decode_forward_test(self, x, input_graph, input_meta, raw_inputs, raw_gt_label, state_dim):
        pred, base_graph = self.decode_head.pre_predict(**x, **input_graph, **input_meta, state_dim=state_dim)
        logits = self.decode_head.forward_test(
            pred=pred, base_graph=base_graph,
            **input_meta, test_cfg=self.test_cfg, h_state=raw_inputs['dynamic']['h_state'])
        
        return logits
    
    def encode_decode(self, inputs, gt_label=None, state_dim=None, **kwargs):
        input_graph, input_meta = self._preprocess(inputs, gt_label, **kwargs)
        x = self.extract_feat(input_graph, input_meta, is_test=True, **kwargs)
        out = self._decode_forward_test(x, input_graph, input_meta, inputs, gt_label, state_dim=state_dim)
        return out
    
    def inference(self, inputs, gt_label=None, is_training=False, state_dim=None, **kwargs):
        output = self.encode_decode(inputs, gt_label=gt_label, is_training=is_training, state_dim=state_dim, **kwargs)
        return output
    
    def _merge_acc(self, pred_rst_list):
        merged_rst = dict()
        merged_rst['pred'] = pred_rst_list[-1]['dynamic']
        merged_rst['acc'] = pred_rst_list[0]['acc']
        for i in range(1, len(pred_rst_list)):
            new_acc = dict()
            for key, val in pred_rst_list[i]['acc'].items():
                new_acc[f"{key}_step{i}"] = val
            merged_rst['acc'].update(new_acc)

        if torch.onnx.is_in_onnx_export():
            return merged_rst
        merged_rst = to_numpy_detach(merged_rst)
        return merged_rst
    
    def simple_test(self, inputs, gt_label=None, **kwargs):
        rollout_size, max_steps = self._rollout_steps(0, 0, self.test_cfg)
        assert rollout_size == 1
        assert kwargs['meta']['frame'] >= 1
        if kwargs['meta']['frame'] <= 1:
            rollout_size += 1
        bs = len(gt_label[0]['vertices'])
        frame_state_dim = gt_label[0]['vertices'][0].shape[1]
        assert frame_state_dim == self.backbone.state_dim
        in_dynamics = inputs['dynamic']
        in_statics = inputs['static']
        assert len(in_dynamics) == max_steps

        pred_rst_list = []
        cur_step = 0
        label_step = max(0, cur_step-self._label_offset())
        cur_dynamic = in_dynamics[label_step]
        cur_label = gt_label[label_step]
        for g_step in range(rollout_size):
            if kwargs['meta']['frame'] == 1 and label_step == 0:
                cloned_label = dict()
                for key, val in cur_label.items():
                    assert isinstance(val, list)
                    cloned_label[key] = [v.clone() for v in val]
                cloned_dynamic = dict()
                for key, val in cur_dynamic.items():
                    assert isinstance(val, list)
                    cloned_dynamic[key] = [v.clone() for v in val]
            else:
                cloned_label = cur_label
                cloned_dynamic = cur_dynamic
            rollout_idx = kwargs['meta']['frame']
            if kwargs['meta']['frame'] == 1:
                rollout_idx = g_step
            pred_dict_i = self.inference(
                dict(dynamic=cloned_dynamic, static=in_statics),
                cloned_label,
                is_training=False, rollout_idx=rollout_idx, state_dim=frame_state_dim, **kwargs)
            pred_dict_i = self._postprocess(dict(dynamic=cloned_dynamic, static=in_statics), pred_dict_i, cloned_label)
            pred_rst_list.append(pred_dict_i)
            cur_step += 1
            label_step = max(0, cur_step-self._label_offset())
            if cur_step >= rollout_size:
                break
            assert gt_label is not None, "Only in train/test can do multi-step pred; Test time cannot"
            next_dynamic = in_dynamics[label_step]
            cur_dynamic = self.rollout_history_input(
                bs, frame_state_dim, pred_dict_i, cloned_dynamic, next_dynamic)
            cur_label = gt_label[label_step]

        pred_rst_list = [pred_rst_list[-1]]
        return self._merge_acc(pred_rst_list)
    
    def _freeze_stages(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return
    
    def train(self, mode=True):
        super().train(mode)
        if self.potential is not None:
            self._freeze_stages(self.potential)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()