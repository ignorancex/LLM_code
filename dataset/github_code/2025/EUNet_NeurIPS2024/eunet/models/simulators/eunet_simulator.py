import torch

from .base import BaseSimulator
from .. import builder
from ..builder import SIMULATORS
from eunet.core import add_prefix, multi_apply
from eunet.datasets.utils import to_numpy_detach
from collections import defaultdict


@SIMULATORS.register_module()
class EUNetSimulator(BaseSimulator):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(EUNetSimulator, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and simulator set pretrained weight'
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.decode_head = builder.build_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _postprocess(self, inputs, pred, gt_label=None):
        return pred
    
    def extract_feat(self, states, templates, faces, f_connectivity, f_connectivity_edges, attr, prev_states, register_norm=False):
        """Extract features from inputs."""
        x = self.backbone(states, templates, faces, f_connectivity, f_connectivity_edges, attr, prev_states=prev_states, register_norm=register_norm)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def encode_decode(self, states, templates, faces, f_connectivity, f_connectivity_edges, attr, prev_states, register_norm=False, **kwargs):
        # Forward
        x = self.extract_feat(states, templates, faces, f_connectivity, f_connectivity_edges, attr, prev_states=prev_states, register_norm=register_norm)
        meta = dict()
        pred = self.decode_head.pre_predict(**x)
        return pred, meta
    
    def _encode_decode_test(self,
                            cur_pred, prev_pred,
                            cur_dynamic, prev_dynamic, prevprev_dynamic,
                            in_statics, **kwargs):
        # Forward
        loss_decode = self.decode_head.forward_test(
            cur_pred, prev_pred,
            cur_dynamic, prev_dynamic, prevprev_dynamic,
            **in_statics)

        return loss_decode

    def _encode_decode_train_loss(self,
                                  cur_pred, prev_pred,
                                  cur_dynamic, prev_dynamic, prevprev_dynamic,
                                  in_static, register_norm, **kwargs):
        losses = dict()
        # Forward
        loss_decode, pred_dynamic = self.decode_head.forward_train(
            cur_pred, prev_pred,
            cur_dynamic, prev_dynamic, prevprev_dynamic,
            **in_static, train_cfg=self.train_cfg, register_norm=register_norm)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, pred_dynamic
    
    def _encode_decode_train_cmp_loss(self,
                                  cur_pred,
                                    cur_dynamic, prev_dynamic,
                                    cur_noised_state, cur_pred_noise,
                                    in_statics,
                                    register_norm, **kwargs):
        losses = dict()
        # Forward
        loss_decode, pred_dynamic = self.decode_head.forward_train_cmp(
            cur_pred,
            cur_dynamic, prev_dynamic,
            cur_noised_state, cur_pred_noise,
            **in_statics, train_cfg=self.train_cfg, register_norm=register_norm)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, pred_dynamic
    
    def forward_train(self, inputs, gt_label, num_epoch=0, num_iter=0, **kwargs):
        rollout_size = 3
        warmup_iters = self.train_cfg.get('warmup', 0)

        in_dynamics = inputs['dynamic']
        in_statics = inputs['static']
        assert in_statics.get('num_iter', None) == None
        assert len(in_dynamics) == rollout_size
        in_statics['num_iter'] = num_iter

        losses = defaultdict(list)
        acc_dict = defaultdict(list)
        # Predict each frame first
        pred_dict_dict = dict()
        pred_noised_dict = dict()
        noised_state_dict = dict()
        
        for cur_step in range(1, rollout_size):
            prev_step = cur_step - 1
            assert prev_step >= 0
            prev_dynamic = in_dynamics[prev_step] # This is for dissipate energy
            cur_dynamic = in_dynamics[cur_step]
            pred_dict_i = self.inference(
                dict(dynamic=cur_dynamic, prev_dynamic=prev_dynamic, static=in_statics),
                is_training=True, register_norm=True, **kwargs)
            pred_dict_i = self._postprocess(dict(dynamic=cur_dynamic, static=in_statics), pred_dict_i)
            pred_dict_dict[cur_step] = pred_dict_i
            
            if cur_step > 1:
                # Preprocess the noise
                orig_state = cur_dynamic.pop('state')
                cur_noise = cur_dynamic['noise']
                noised_state = [o_s + noise*h_mask for o_s, noise, h_mask in zip(orig_state, cur_noise, in_statics['hop_mask'])]
                noised_state_dict[cur_step] = noised_state
                cur_dynamic['state'] = noised_state

                pred_noise_i = self.inference(
                    dict(dynamic=cur_dynamic, prev_dynamic=prev_dynamic, static=in_statics),
                    is_training=True, register_norm=False, **kwargs)
                pred_noise_i = self._postprocess(dict(dynamic=cur_dynamic, static=in_statics), pred_noise_i)
                pred_noised_dict[cur_step] = pred_noise_i
                # Recover the state
                cur_dynamic['state'] = orig_state
            else:
                pred_noised_dict[cur_step] = None
                noised_state_dict[cur_step] = None
        
        # Supervised energy
        init_step = 2
        for cur_step_idx in range(init_step, rollout_size):
            # Extra one to skip the initialization frame
            prev_step_idx = cur_step_idx - 1
            prevprev_step_idx = prev_step_idx - 1
            assert prevprev_step_idx >= 0

            prevprev_dynamic = in_dynamics[prevprev_step_idx]
            prev_dynamic = in_dynamics[prev_step_idx]
            cur_dynamic = in_dynamics[cur_step_idx]

            prev_pred = pred_dict_dict[prev_step_idx]
            cur_pred = pred_dict_dict[cur_step_idx]

            losses_i, pred_dynamic_i = self._encode_decode_train_loss(
                cur_pred, prev_pred,
                cur_dynamic, prev_dynamic, prevprev_dynamic,
                in_statics,
                register_norm=True,
                **kwargs)

            # Merge loss
            for key in losses_i.keys():
                l_key = f"{key}"
                if key.startswith('decode.loss'):
                    # Sum loss
                    losses[l_key].append(losses_i[key])
                else:
                    acc_dict[l_key].append(losses_i[key])
        
        # Contrastive loss
        cmp_init = 2
        for cur_step_idx in range(cmp_init, rollout_size):
            prev_step_idx = cur_step_idx - 1
            prevprev_step_idx = prev_step_idx - 1
            assert prevprev_step_idx >= 0

            prev_dynamic = in_dynamics[prev_step_idx]
            cur_dynamic = in_dynamics[cur_step_idx]
            cur_pred = pred_dict_dict[cur_step_idx]

            cur_noised_state = noised_state_dict[cur_step_idx]
            cur_pred_noise = pred_noised_dict[cur_step_idx]

            losses_i, pred_dynamic_i = self._encode_decode_train_cmp_loss(
                cur_pred,
                cur_dynamic, prev_dynamic,
                cur_noised_state, cur_pred_noise,
                in_statics,
                register_norm=False,
                **kwargs)

            # Merge loss
            for key in losses_i.keys():
                l_key = f"{key}_cmp"
                if key.startswith('decode.loss'):
                    # Sum loss
                    losses[l_key].append(losses_i[key])
                else:
                    acc_dict[l_key].append(losses_i[key])
            
        # reduce losses
        losses_rst = dict()
        for key, val in losses.items():
            losses_rst[key] = torch.mean(torch.stack(val))
            if warmup_iters > num_iter:
                losses_rst[key] *= 0
        for key, val in acc_dict.items():
            losses_rst[key] = torch.mean(torch.stack(val))

        return losses_rst

    def _preprocess(self, inputs, **kwargs):
        return inputs

    def inference(self, inputs, is_training=False, register_norm=False, **kwargs):
        """Inference with slide/whole style.
        """
        # Preprocess data
        inputs = self._preprocess(inputs, **kwargs)
        states = [d_state[:, :3] for d_state in inputs['dynamic']['state']]
        prev_states = [p_states[:, :3] for p_states in inputs['prev_dynamic']['state']]
        f_connectivity = inputs['static']['f_connectivity']
        f_connectivity_edges = inputs['static']['f_connectivity_edges']
        templates = inputs['static']['templates']
        faces = inputs['static']['faces']
        attr = inputs['static']['attr']

        rst = dict()
        output, meta = multi_apply(
            self.encode_decode,
            states, templates, faces, f_connectivity, f_connectivity_edges, attr, prev_states, register_norm=register_norm)
        rst['pred'] = output
        rst['pred_meta'] = meta

        return rst
    
    def simple_test(self, inputs, gt_label=None, **kwargs):
        """Simple test with single image."""
        rollout_size = 3
        in_dynamics = inputs['dynamic']
        in_statics = inputs['static']
        assert len(in_dynamics) == rollout_size

        losses = defaultdict(list)
        acc_dict = defaultdict(list)
        # Predict each frame first
        pred_dict_dict = dict()
        for cur_step in range(1, rollout_size):
            prev_step = cur_step - 1
            assert prev_step >= 0
            prev_dynamic = in_dynamics[prev_step] # This is for dissipate energy
            cur_dynamic = in_dynamics[cur_step]
            pred_dict_i = self.inference(
                dict(dynamic=cur_dynamic, prev_dynamic=prev_dynamic, static=in_statics),
                is_training=False, register_norm=False, **kwargs)
            pred_dict_i = self._postprocess(dict(dynamic=cur_dynamic, static=in_statics), pred_dict_i)
            pred_dict_dict[cur_step] = pred_dict_i

        init_step = 2
        for cur_step_idx in range(init_step, rollout_size):
            # Extra one to skip the initialization frame
            prev_step_idx = cur_step_idx - 1
            prevprev_step_idx = prev_step_idx - 1
            assert prevprev_step_idx >= 0

            prevprev_dynamic = in_dynamics[prevprev_step_idx]
            prev_dynamic = in_dynamics[prev_step_idx]
            cur_dynamic = in_dynamics[cur_step_idx]

            prev_pred = pred_dict_dict[prev_step_idx]
            cur_pred = pred_dict_dict[cur_step_idx]

            losses_i, pred_dynamic_i = self._encode_decode_test(
                cur_pred, prev_pred,
                cur_dynamic, prev_dynamic, prevprev_dynamic,
                in_statics,
                **kwargs)
            # Merge loss
            for key in losses_i['acc'].keys():
                acc_dict[key].append(losses_i['acc'][key])

        rst_acc_dict = dict()
        for key, val in acc_dict.items():
            rst_acc_dict[key] = torch.mean(torch.stack(val))
        assert min(list(pred_dict_dict.keys())) == 1
        merged_rst = dict(acc=rst_acc_dict, pred=pred_dict_dict[1]['pred']) # minimum step start from 1, using 0 for dissipate
        return self._merge_acc(merged_rst)
    
    def _merge_acc(self, acc_dict):
        # Merge acc; pred cannot merge and no use actually
        merged_rst = acc_dict

        if torch.onnx.is_in_onnx_export():
            return merged_rst
        merged_rst = to_numpy_detach(merged_rst)
        return merged_rst
