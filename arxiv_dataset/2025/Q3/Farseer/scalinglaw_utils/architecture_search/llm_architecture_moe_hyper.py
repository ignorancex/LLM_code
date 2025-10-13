
import numpy as np
import torch
from scalinglaw_utils.architecture_search.ssm_layer_allocate.mamba_hybrid_layer_allocation import allocate_layers
from scalinglaw_utils.architecture_search.llm_architecture_base import BaseLLMArchitecture, ParaAndCompute
from scalinglaw_utils.architecture_search.llm_architecture_base_hyper import HyperMamba, HyperMambaGQA
import torch.nn.functional as F

class HyperMambaMoE(HyperMamba):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, ssm_expand=2, ssm_d_state=128, ngroups=8, ssm_head_dim=128, chunck_size=128, expert_num=0, top_k=0, dim_dynamic_ffn=0, dim_share_ffn=0, moe_layer=0):
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num, hybrid_attention_ratio, hybrid_mlp_ratio, ssm_expand, ssm_d_state, ssm_head_dim, ngroups, chunck_size)
        self.E = float(expert_num)
        self.K = float(top_k)
        self.ds = float(dim_share_ffn)
        self.de = float(dim_dynamic_ffn)
        self.Le = float(moe_layer)
        kxi = 1e-06
        self.beta = (((self.K * self.de) + self.ds) / (self.d_model + kxi))
        self.mu = (((self.E * self.de) + self.ds) / (self.d_model + kxi))
        self.delta = (((4 + (3 * self.alpha)) * (self.L - self.Le)) / self.L)
        self.sparsity = (((4 + (3 * self.beta)) + self.delta) / ((4 + (3 * self.mu)) + self.delta))
        self.rho = (self.ds / ((self.K * self.de) + self.ds))
        self.zeta = (self.d_model / self.L)
        self.att_ratio = (4 / ((4 + (3 * self.mu)) + (((4 + (3 * self.alpha)) * (self.L - self.Le)) / self.L)))
        self.p_and_c_act = ParaAndCompute(0, 0)

    def reset(self):
        self.para_and_compute = ParaAndCompute(0, 0)
        self.p_and_c_non_vocab = ParaAndCompute(0, 0)
        self.p_and_c_act = ParaAndCompute(0, 0)

    def _base_str(self):
        base_str = super()._base_str()
        return (base_str + ('E:%d, K:%d, De:%d, Ds:%d, Le:%d, ' % (self.E, self.K, self.de, self.ds, self.Le)))

    def __str__(self):
        return (((((((((((f"mm/ma :{((self.p_and_c_mamba2.para / self.layer_types.count('M')) / (self.p_and_c_att.para / self.layer_types.count('*')))}, " + self._base_str()) + self._att_str()) + self._ratio_str()) + ', non_emb para:') + str(self.p_and_c_non_vocab)) + ', act para:') + str(self.p_and_c_act)) + ', para:') + str(self.para_and_compute)) + ' Hybrid allocation: ') + ''.join(self.layer_types))

    def __eq__(self, other):
        res = super().__eq__(other)
        res = (res and (self.E == other.E) and (self.K == other.K) and (self.ds == other.ds) and (self.de == other.de) and (self.Le == other.Le))
        return res

    def __hash__(self):
        return hash((self.S, self.d_model, self.d_ffn, self.L, self.V, self.head_dim, self.head_num, self.E, self.K, self.ds, self.de, self.Le, ''.join(self.layer_types)))

    def parameter_count(self):
        super().parameter_count()
        self.p_and_c_act = ((((self.p_and_c_act + self.p_and_c_ffn_act) + self.p_and_c_other) + self.p_and_c_att) + self.p_and_c_mamba2)
        self.para_and_compute.to_float()
        self.p_and_c_act.to_float()
        return self.para_and_compute.to_float()

    def count_ffn(self):
        para = (((3.0 * self.d_model) * self.d_ffn) * (self.layer_types.count('-') - self.Le))
        para_router = ((self.d_model * self.E) * self.Le)
        para_share = (((3.0 * self.d_model) * self.ds) * self.Le)
        para_moe_fnn = ((((3.0 * self.d_model) * self.de) * self.E) * self.Le)
        para_moe_act = ((((3.0 * self.d_model) * self.de) * self.K) * self.Le)
        para_all = (((para + para_share) + para_moe_fnn) + para_router)
        para_act = (((para + para_share) + para_moe_act) + para_router)
        self.p_and_c_ffn = ParaAndCompute(para_all, (6.0 * para_act))
        self.p_and_c_ffn_act = ParaAndCompute(para_act, (6.0 * para_act))
        return self.p_and_c_ffn

    def get_kvcache_size(self, sequence_length: float):
        return ((((((sequence_length * self.layer_types.count('*')) * self.head_dim) * self.head_num) * 2.0) + ((self.layer_types.count('M') * int(((self.ssm_expand * self.d_model) + (2 * self.ssm_d_state)))) * 4)) + ((self.layer_types.count('M') * int((self.ssm_expand * self.d_model))) * self.ssm_d_state))

class HyperMambaMoEMFA2(HyperMambaMoE):

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def _att_compute(self):
        attention = ((((4.0 * self.S) * self.head_dim) * self.head_num) * self.layer_types.count('*'))
        softmax = (((3.0 * self.S) * self.head_num) * self.layer_types.count('*'))
        return (attention, softmax)

    def _qkvo_compute(self):
        wo = (((self.d_model * self.head_num) * self.head_dim) * self.layer_types.count('*'))
        qkv_share = (((self.d_model * 3) * self.head_dim) * self.layer_types.count('*'))
        wq = ((self.head_num * (self.head_dim ** 2)) * self.layer_types.count('*'))
        return ((wo + qkv_share) + wq)

    def get_kvcache_size(self, sequence_length: float):
        return (((((sequence_length * self.layer_types.count('*')) * self.head_dim) * 2.0) + ((self.layer_types.count('M') * int((self.ssm_expand * self.d_model))) * 4)) + ((self.layer_types.count('M') * int((self.ssm_expand * self.d_model))) * self.ssm_d_state))

class HyperMambaMoEGQA(HyperMambaGQA):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, ssm_expand=2, ssm_d_state=128, ngroups=8, ssm_head_dim=128, chunck_size=128, expert_num=0, top_k=0, dim_dynamic_ffn=0, dim_share_ffn=0, moe_layer=0, group=2):
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num, hybrid_attention_ratio, hybrid_mlp_ratio, ssm_expand, ssm_d_state, ssm_head_dim, ngroups, chunck_size, group=group)
        self.E = float(expert_num)
        self.K = float(top_k)
        self.ds = float(dim_share_ffn)
        self.de = float(dim_dynamic_ffn)
        self.Le = float(moe_layer)
        kxi = 1e-06
        self.beta = (((self.K * self.de) + self.ds) / (self.d_model + kxi))
        self.mu = (((self.E * self.de) + self.ds) / (self.d_model + kxi))
        self.delta = (((4 + (3 * self.alpha)) * (self.L - self.Le)) / self.L)
        self.sparsity = (((4 + (3 * self.beta)) + self.delta) / ((4 + (3 * self.mu)) + self.delta))
        self.rho = (self.ds / ((self.K * self.de) + self.ds))
        self.zeta = (self.d_model / self.L)
        self.att_ratio = (4 / ((4 + (3 * self.mu)) + (((4 + (3 * self.alpha)) * (self.L - self.Le)) / self.L)))
        self.p_and_c_act = ParaAndCompute(0, 0)

    def reset(self):
        self.para_and_compute = ParaAndCompute(0, 0)
        self.p_and_c_non_vocab = ParaAndCompute(0, 0)
        self.p_and_c_act = ParaAndCompute(0, 0)

    def _base_str(self):
        base_str = super()._base_str()
        return (base_str + ('E:%d, K:%d, De:%d, Ds:%d, Le:%d, ' % (self.E, self.K, self.de, self.ds, self.Le)))

    def __str__(self):
        self.parameter_count()
        return ((((((((((self._base_str() + self._att_str()) + self._ratio_str()) + ', non_emb para:') + str(self.p_and_c_non_vocab)) + ', act para:') + str(self.p_and_c_act)) + ', para:') + str(self.para_and_compute)) + ' Hybrid allocation: ') + ''.join(self.layer_types))

    def __eq__(self, other):
        res = super().__eq__(other)
        res = (res and (self.E == other.E) and (self.K == other.K) and (self.ds == other.ds) and (self.de == other.de) and (self.Le == other.Le))
        return res

    def __hash__(self):
        return hash((self.S, self.d_model, self.d_ffn, self.L, self.V, self.head_dim, self.head_num, self.E, self.K, self.ds, self.de, self.Le, ''.join(self.layer_types)))

    def parameter_count(self):
        super().parameter_count()
        self.p_and_c_act = ((((self.p_and_c_act + self.p_and_c_ffn_act) + self.p_and_c_other) + self.p_and_c_att) + self.p_and_c_mamba2)
        self.para_and_compute.to_float()
        self.p_and_c_act.to_float()
        return self.para_and_compute.to_float()

    def count_ffn(self):
        para = (((3.0 * self.d_model) * self.d_ffn) * (self.layer_types.count('-') - self.Le))
        para_router = ((self.d_model * self.E) * self.Le)
        para_share = (((3.0 * self.d_model) * self.ds) * self.Le)
        para_moe_fnn = ((((3.0 * self.d_model) * self.de) * self.E) * self.Le)
        para_moe_act = ((((3.0 * self.d_model) * self.de) * self.K) * self.Le)
        para_all = (((para + para_share) + para_moe_fnn) + para_router)
        para_act = (((para + para_share) + para_moe_act) + para_router)
        self.p_and_c_ffn = ParaAndCompute(para_all, (6.0 * para_act))
        self.p_and_c_ffn_act = ParaAndCompute(para_act, (6.0 * para_act))
        return self.p_and_c_ffn

    def get_kvcache_size(self, sequence_length: float):
        return ((((((sequence_length * self.layer_types.count('*')) * self.head_dim) * self.group) * 2.0) + ((self.layer_types.count('M') * int((self.ssm_expand * self.d_model))) * 4)) + ((self.layer_types.count('M') * int((self.ssm_expand * self.d_model))) * self.ssm_d_state))

def search_hyper_ssm_atch_moe(para, emb_ratio, wo_emb_ratio, bound1, bound2, tight_bound, with_emb=True, alpha_low=1.0, alpha_high=100.0, s_fix=2048, head_dim=128, print_out=True, hybrid_attention_ratio=0.2, hybrid_mlp_ratio=0.4, consider_compute=False, error_bound=0.005, expert_num=0, top_k=0, dim_dynamic_ffn=0, dim_share_ffn=0, moe_layer=0):
    print((('Searching ' + '{:.5e}'.format(para)) + ('===' * 50)))
    v_fix = 65536
    (d_model_start, d_model_end) = (head_dim, 20480)
    (l_start, l_end) = (3, 128)
    (res_double, res_tight1, res_tight2) = ([], [], [])
    bounds = [(emb_ratio - bound1), (emb_ratio + bound1), (wo_emb_ratio - bound2), (wo_emb_ratio + bound2), (emb_ratio - tight_bound), (emb_ratio + tight_bound), (wo_emb_ratio - tight_bound), (wo_emb_ratio + tight_bound)]
    for layer in range(l_start, l_end):
        for d in range(d_model_start, d_model_end, 64):
            for head_num in [8, 16]:
                for ffn_expand in [2, 4, 8]:
                    for moe_expand in [2, 4, 8, 16]:
                        head_dim = (d // head_num)
                        if (head_dim not in [64, 128, 192, 256]):
                            continue
                        d = (head_dim * head_num)
                        dff = (d * ffn_expand)
                        dim_dynamic_ffn = (d // moe_expand)
                        dim_share_ffn = (2 * dim_dynamic_ffn)
                        moe_layer = ((layer // 2) - 1)
                        model = HyperMambaMoE(s_len=s_fix, dim_model=d, dim_dff=dff, layer=layer, head_dim=head_dim, v_dim=v_fix, hybrid_attention_ratio=hybrid_attention_ratio, hybrid_mlp_ratio=hybrid_mlp_ratio, expert_num=expert_num, top_k=top_k, dim_dynamic_ffn=dim_dynamic_ffn, dim_share_ffn=dim_share_ffn, moe_layer=moe_layer)
                        model.parameter_count()
                        if (not consider_compute):
                            if (((1 - error_bound) <= (model.p_and_c_non_vocab.para / para)) and ((model.p_and_c_non_vocab.para / para) <= (1 + error_bound))):
                                model.reset()
                                res_tight2.append(model)
                        elif (((1 - error_bound) <= (model.para_and_compute.compute / para)) and ((model.para_and_compute.compute / para) <= (1 + error_bound))):
                            model.reset()
                            res_tight2.append(model)
    if print_out:
        print('Ratio tight')
        for r in (res_tight1 if with_emb else res_tight2):
            print(r)
    return (res_tight1 if with_emb else res_tight2)
if (__name__ == '__main__'):
    dim = 2048
    depth = 16
    ffn_dim = 8192
    n = ((dim * depth) * ((4 * dim) + (3 * ffn_dim)))
    test = HyperMamba(s_len=2048, dim_model=960, dim_dff=9368, layer=14, head_dim=64, v_dim=65536, hybrid_attention_ratio=0.5, hybrid_mlp_ratio=0.5)
    print(str(test))
