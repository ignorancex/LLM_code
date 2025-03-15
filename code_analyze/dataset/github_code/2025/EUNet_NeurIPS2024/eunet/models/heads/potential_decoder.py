import torch

from ..builder import HEADS
from .sim_head import SimHead
from eunet.models.utils import FFN
from eunet.core import multi_apply

from eunet.models.utils import Normalizer
from eunet.models.utils.energy_utils import euler_energy, comp_energy
from eunet.datasets.utils.metacloth import TYPE_NAMES


@HEADS.register_module()
class PotentialDecoder(SimHead):
    def __init__(self,
                 out_channels=1, # 1 for scalar
                 in_channels=128,
                 position_dim=3,
                 dt=1/30,
                 init_cfg=None,
                 eps=1e-7,
                 act_cfg=dict(type='SiLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 norm_acc_steps=10000,
                 dissipate_sigma=0.5,
                 *args,
                 **kwargs):
        super(PotentialDecoder, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.position_dim = position_dim
        self.dt = dt
        self.eps = eps
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={out_channels} must be a positive integer')

        self.norm_acc_steps = norm_acc_steps
        self._init_energy_head()

        self.dissipate_sigma = dissipate_sigma
        if dissipate_sigma is not None:
            self._init_dissipate_energy_head()

    def _init_energy_head(self):
        energy_out_dim = self.out_channels
        energy_act_cfg = self.act_cfg
        self.energy_head = FFN(
            [self.in_channels, self.in_channels//2, self.in_channels//4, energy_out_dim],
            act_cfg=energy_act_cfg,
            bias=True,
            final_act=False,)
        self.energy_normalizer = Normalizer(size=1, max_accumulations=self.norm_acc_steps)

    def _init_dissipate_energy_head(self):
        energy_act_cfg = self.act_cfg
        self.dissipate_energy_head = FFN(
            [self.in_channels, self.in_channels//2, self.in_channels//4, self.out_channels],
            act_cfg=energy_act_cfg,
            bias=False,
            last_bias=False,
            final_act=False,)
        self.dissipate_energy_head_final_act = lambda x: x**2 # Make sure the dissipation must be non-negative

    def init_weights(self):
        super(PotentialDecoder, self).init_weights()

    def evaluate(self, pred, gt_label, **kwargs):
        acc_dict = self.accuracy(pred, gt_label, **kwargs)
        return acc_dict
    
    def pre_predict(self, energy_emb, face_area=None, **kwargs):
        pred = self.predict(energy_emb, face_area=face_area, **kwargs)
        
        return pred
    
    def predict(self, energy_emb, face_area=None, cur_delta_l2=None, coeff_k=None, **kwargs):
        if self.dissipate_sigma is None:
            energy_out = self.energy_head(energy_emb)
        else:
            potential_energy_emb = energy_emb[:, :self.in_channels]
            dissipate_energy_emb = energy_emb[:, self.in_channels:]
            assert dissipate_energy_emb.shape[-1] == self.in_channels
            p_energy_out = self.energy_head(potential_energy_emb)
            d_energy_out = self.dissipate_energy_head_final_act(self.dissipate_energy_head(dissipate_energy_emb))
            assert face_area is not None
            d_energy_out *= face_area
            energy_sum = p_energy_out + d_energy_out
            energy_out = torch.cat([energy_sum, d_energy_out], dim=-1)
        # The absolute value is meaningless
        energy_denormed = self.energy_normalizer.inverse(energy_out, override_mean=0.0)
        
        assert not torch.isnan(energy_denormed).any()
        return energy_denormed

    def forward_train_cmp(self,
                        cur_pred,
                        cur_dynamic, prev_dynamic,
                        cur_noised_state, cur_pred_noise,
                        mass, hop_mask, f_connectivity_edges, first_neighbor_mask, **kwargs):
        losses = dict()
        cur_total_e, noised_total_e = multi_apply(
            comp_energy,
            cur_pred['pred'], cur_dynamic['state'], prev_dynamic['state'],
            cur_noised_state, cur_pred_noise['pred'], cur_dynamic['gravity'],
            mass, hop_mask, f_connectivity_edges, first_neighbor_mask,
            dt=self.dt)

        noised_losses = self.loss(
            # cmp loss
            noised_total_e, cur_total_e,
            type_names=TYPE_NAMES,
            term_filter=['elcmp'], cal_acc=True, hop_mask=hop_mask, **kwargs)
        losses.update(noised_losses)

        return losses, None

    def forward_train(self,
            cur_pred, prev_pred,
            cur_dynamic, prev_dynamic, prevprev_dynamic,
            mass, register_norm,
            vert_mask=None, **kwargs):
        '''
            trans must be 0, must use global axis
        '''
        bs = len(cur_pred['pred'])
        gravity = cur_dynamic['gravity']
        losses = dict()

        gt_label_elenergy, pred_elenergy = multi_apply(
            euler_energy,
            cur_dynamic['state'], prev_dynamic['state'], prevprev_dynamic['state'],
            [cp[:, :self.out_channels] for cp in cur_pred['pred']],
            [pp[:, :self.out_channels] for pp in prev_pred['pred']],
            gravity, mass,
            [None]*bs,
            [pp[:, self.out_channels:] for pp in prev_pred['pred']] if self.dissipate_sigma is not None else [None]*bs,
            dt=self.dt, f_ext=None)
        loss_pred_elenergy = pred_elenergy
        loss_gt_elenergy = gt_label_elenergy
        elenergy_losses = self.loss(
            loss_pred_elenergy, loss_gt_elenergy,
            type_names=TYPE_NAMES,
            vert_mask=None,
            term_filter=['elenergy'], cal_acc=True, **kwargs)
        losses.update(elenergy_losses)

        if register_norm:
            # Temparorily normalize energy output by exisitng ones
            _, delta_energy = multi_apply(
                euler_energy,
                cur_dynamic['state'], prev_dynamic['state'], prevprev_dynamic['state'],
                [cp[:, :self.out_channels] for cp in cur_pred['pred']],
                [pp[:, :self.out_channels] for pp in prev_pred['pred']],
                gravity, mass,
                [None]*bs, # Placeholder
                [pp[:, self.out_channels:] for pp in prev_pred['pred']] if self.dissipate_sigma is not None else [None]*bs,
                dt=self.dt, f_ext=None)
            _ = self.energy_normalizer(torch.cat(delta_energy, dim=0))
        return losses, None
    
    def forward_test(self, *args, **kwargs):
        return self.simple_test(*args, **kwargs)

    def simple_test(self,
                    cur_pred, prev_pred,
                    cur_dynamic, prev_dynamic, prevprev_dynamic,
                    mass, vert_mask=None, **kwargs):
        bs = len(cur_pred['pred'])
        rst = dict()
        acc_dict = dict()
        gravity = cur_dynamic['gravity']

        gt_label_elenergy, pred_elenergy = multi_apply(
            euler_energy,
            cur_dynamic['state'], prev_dynamic['state'], prevprev_dynamic['state'],
            [cp[:, :self.out_channels] for cp in cur_pred['pred']],
            [pp[:, :self.out_channels] for pp in prev_pred['pred']],
            gravity, mass,
            [None]*bs,
            [pp[:, self.out_channels:] for pp in prev_pred['pred']] if self.dissipate_sigma is not None else [None]*bs,
            dt=self.dt, f_ext=None)
        accelenergy_dict = self.evaluate(
            pred_elenergy, gt_label_elenergy,
            type_names=TYPE_NAMES,
            vert_mask=None,
            term_filter=['elenergy'], overwrite_align=True, **kwargs)
        acc_dict.update(accelenergy_dict)

        rst.update(dict(acc=acc_dict)) 

        return rst, None
    
    def train(self, mode=True):
        super().train(mode)