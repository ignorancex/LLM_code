
import numpy as np
import torch
from scalinglaw_utils.architecture_search.ssm_layer_allocate.mamba_hybrid_layer_allocation import allocate_layers
from scalinglaw_utils.architecture_search.llm_architecture_base import BaseLLMArchitecture, ParaAndCompute, GQAArchitecture
import torch.nn.functional as F

class HyperMamba(BaseLLMArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, ssm_expand=2, ssm_d_state=128, ssm_head_dim=128, ngroups=8, chunck_size=128):
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num)
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.layer_types = allocate_layers(total_layers_count=layer, target_mlp_ratio=self.hybrid_mlp_ratio, target_attention_ratio=self.hybrid_attention_ratio)
        self.ssm_d_state = ssm_d_state
        self.ssm_expand = ssm_expand
        self.ssm_chunk_size = chunck_size
        self.ssm_head_dim = ssm_head_dim
        self.ngroups = ngroups

    def _base_str(self):
        base_str = super()._base_str()
        base_str = (base_str + ('ssm_d:%d, ssm_e:%0.20f, ssm_chunk_size:%d, ssm_head_dim:%d, ngroups:%d, nhead:%d ' % (self.ssm_d_state, self.ssm_expand, self.ssm_chunk_size, self.ssm_head_dim, self.ngroups, (int((self.ssm_expand * self.d_model)) / self.ssm_head_dim))))
        return base_str

    def __str__(self):
        return ((((((((self._base_str() + self._att_str()) + self._ratio_str()) + '  para:') + str(self.para_and_compute)) + ', non_emb para:') + str(self.p_and_c_non_vocab)) + ' Hybrid allocation: ') + ''.join(self.layer_types))

    def reset(self):
        self.para_and_compute = ParaAndCompute(0, 0)
        self.p_and_c_non_vocab = ParaAndCompute(0, 0)

    def get_mamba_paras_compute(self):

        def ssd_flops(T, Q, P, N):
            center_blocks_sma_compute = ((((T * Q) * N) + ((T * Q) * Q)) + ((T * P) * N))
            b_compute = ((T * N) * P)
            a_compute = (((T * N) * P) / Q)
            c_compute = ((T * P) * N)
            return (((center_blocks_sma_compute + b_compute) + a_compute) + c_compute)

        def conv_flops_counter(batch_size=1, seq_length=2048, kernel_size=4, in_channels=128, out_channels=128, groups=128, extra_per_position_flops=0, bias=True):
            output_dims = [seq_length]
            kernel_dims = kernel_size
            in_channels = in_channels
            out_channels = out_channels
            groups = groups
            filters_per_channel = (out_channels // groups)
            conv_per_position_flops = (int(np.prod(kernel_dims, dtype=np.int64)) * ((in_channels * filters_per_channel) + extra_per_position_flops))
            active_elements_count = (batch_size * int(np.prod(output_dims, dtype=np.int64)))
            overall_conv_flops = (conv_per_position_flops * active_elements_count)
            bias_flops = 0
            if bias:
                bias_flops = (out_channels * active_elements_count)
            overall_flops = (overall_conv_flops + bias_flops)
            return overall_flops
        d_inner = int((self.ssm_expand * self.d_model))
        ngroups = self.ngroups
        d_state = self.ssm_d_state
        nheads = (d_inner // self.ssm_head_dim)
        conv_dim = (d_inner + ((2 * ngroups) * d_state))
        in_proj_paras = (self.d_model * (((d_inner * 2) + ((2 * ngroups) * d_state)) + nheads))
        out_proj_paras = (d_inner * self.d_model)
        conv1d_paras = (((conv_dim * (conv_dim / conv_dim)) * 4) + conv_dim)
        conv1d_compute = (conv_flops_counter(batch_size=1, seq_length=self.S, kernel_size=4, in_channels=conv_dim, out_channels=conv_dim, groups=conv_dim) / self.S)
        dt_bias_paras = nheads
        A_log_paras = nheads
        D_paras = nheads
        rmsnorm_paras = d_inner
        rmsnorm_paras_compute = (7.5 * rmsnorm_paras)
        scan_flops_2 = (ssd_flops(T=self.S, Q=self.ssm_chunk_size, P=d_inner, N=(ngroups * d_state)) / self.S)
        linear_flops = (in_proj_paras + out_proj_paras)
        return (((((((in_proj_paras + out_proj_paras) + conv1d_paras) + dt_bias_paras) + A_log_paras) + D_paras) + rmsnorm_paras), ((((scan_flops_2 * 2) + (2 * linear_flops)) + (conv1d_compute * 2)) + rmsnorm_paras_compute), conv1d_compute)

    def count_mamba2(self):
        total_params = 0
        total_compute = 0
        total_conv1d_compute = 0
        for i in self.layer_types:
            if (i == 'M'):
                (params, compute, conv1d_compute) = self.get_mamba_paras_compute()
                total_compute += compute
                total_params += params
                total_conv1d_compute += conv1d_compute
        self.p_and_c_mamba2 = ParaAndCompute(total_params, (3.0 * total_compute))
        self.total_conv1d_compute = (total_conv1d_compute * 3.0)
        return self.p_and_c_mamba2

    def _att_compute(self):
        attention = ((((4.0 * self.S) * self.head_dim) * self.head_num) * self.layer_types.count('*'))
        softmax = (((3.0 * self.S) * self.head_num) * self.layer_types.count('*'))
        return (attention, softmax)

    def _qkvo_compute(self):
        return ((((4.0 * self.d_model) * self.head_num) * self.head_dim) * self.layer_types.count('*'))

    def count_ffn(self):
        para = (((3.0 * self.d_model) * self.d_ffn) * self.layer_types.count('-'))
        self.p_and_c_ffn = ParaAndCompute(para, (6.0 * para))
        return self.p_and_c_ffn

    def count_other(self):
        para = (self.d_model * self.L)
        self.p_and_c_other = ParaAndCompute(para, ((7.5 * 3.0) * para))
        return self.p_and_c_other

    def parameter_count(self):
        self.p_and_c_non_vocab += self.count_attention()
        self.p_and_c_non_vocab += self.count_ffn()
        self.p_and_c_non_vocab += self.count_other()
        self.p_and_c_non_vocab += self.count_mamba2()
        self.p_and_c_non_vocab.to_float()
        self.para_and_compute = ((self.para_and_compute + self.p_and_c_non_vocab) + self.count_vocab())
        self.para_and_compute.to_float()
        return self.para_and_compute.to_float()

class HyperMambaGQA(GQAArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, ssm_expand=2, ssm_d_state=128, ssm_head_dim=128, ngroups=8, chunck_size=128, group=2):
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num, group=group)
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.layer_types = allocate_layers(total_layers_count=layer, target_mlp_ratio=self.hybrid_mlp_ratio, target_attention_ratio=self.hybrid_attention_ratio)
        self.ssm_d_state = ssm_d_state
        self.ssm_expand = ssm_expand
        self.ssm_chunk_size = chunck_size
        self.ssm_head_dim = ssm_head_dim
        self.ngroups = ngroups

    def _base_str(self):
        base_str = super()._base_str()
        base_str = (base_str + ('ssm_d:%d, ssm_e:%0.20f, ssm_chunk_size:%d, ssm_head_dim:%d, ngroups:%d, nhead:%d ' % (self.ssm_d_state, self.ssm_expand, self.ssm_chunk_size, self.ssm_head_dim, self.ngroups, (int((self.ssm_expand * self.d_model)) / self.ssm_head_dim))))
        return base_str

    def __str__(self):
        self.parameter_count()
        return ((((((((self._base_str() + self._att_str()) + self._ratio_str()) + '  para:') + str(self.para_and_compute)) + ', non_emb para:') + str(self.p_and_c_non_vocab)) + ' Hybrid allocation: ') + ''.join(self.layer_types))

    def reset(self):
        self.para_and_compute = ParaAndCompute(0, 0)
        self.p_and_c_non_vocab = ParaAndCompute(0, 0)

    def get_mamba_paras_compute(self):

        def ssd_flops(T, Q, P, N):
            center_blocks_sma_compute = ((((T * Q) * N) + ((T * Q) * Q)) + ((T * P) * N))
            b_compute = ((T * N) * P)
            a_compute = (((T * N) * P) / Q)
            c_compute = ((T * P) * N)
            return (((center_blocks_sma_compute + b_compute) + a_compute) + c_compute)

        def conv_flops_counter(batch_size=1, seq_length=2048, kernel_size=4, in_channels=128, out_channels=128, groups=128, extra_per_position_flops=0, bias=True):
            output_dims = [seq_length]
            kernel_dims = kernel_size
            in_channels = in_channels
            out_channels = out_channels
            groups = groups
            filters_per_channel = (out_channels // groups)
            conv_per_position_flops = (int(np.prod(kernel_dims, dtype=np.int64)) * ((in_channels * filters_per_channel) + extra_per_position_flops))
            active_elements_count = (batch_size * int(np.prod(output_dims, dtype=np.int64)))
            overall_conv_flops = (conv_per_position_flops * active_elements_count)
            bias_flops = 0
            if bias:
                bias_flops = (out_channels * active_elements_count)
            overall_flops = (overall_conv_flops + bias_flops)
            return overall_flops
        d_inner = int((self.ssm_expand * self.d_model))
        ngroups = self.ngroups
        d_state = self.ssm_d_state
        nheads = (d_inner // self.ssm_head_dim)
        conv_dim = (d_inner + ((2 * ngroups) * d_state))
        in_proj_paras = (self.d_model * (((d_inner * 2) + ((2 * ngroups) * d_state)) + nheads))
        out_proj_paras = (d_inner * self.d_model)
        conv1d_paras = (((conv_dim * (conv_dim / conv_dim)) * 4) + conv_dim)
        conv1d_compute = (conv_flops_counter(batch_size=1, seq_length=self.S, kernel_size=4, in_channels=conv_dim, out_channels=conv_dim, groups=conv_dim) / self.S)
        dt_bias_paras = nheads
        A_log_paras = nheads
        D_paras = nheads
        rmsnorm_paras = d_inner
        rmsnorm_paras_compute = (7.5 * rmsnorm_paras)
        scan_flops_2 = (ssd_flops(T=self.S, Q=self.ssm_chunk_size, P=d_inner, N=(ngroups * d_state)) / self.S)
        linear_flops = (in_proj_paras + out_proj_paras)
        return (((((((in_proj_paras + out_proj_paras) + conv1d_paras) + dt_bias_paras) + A_log_paras) + D_paras) + rmsnorm_paras), ((((scan_flops_2 * 2) + (2 * linear_flops)) + (conv1d_compute * 2)) + rmsnorm_paras_compute), conv1d_compute)

    def count_mamba2(self):
        total_params = 0
        total_compute = 0
        total_conv1d_compute = 0
        for i in self.layer_types:
            if (i == 'M'):
                (params, compute, conv1d_compute) = self.get_mamba_paras_compute()
                total_compute += compute
                total_params += params
                total_conv1d_compute += conv1d_compute
        self.p_and_c_mamba2 = ParaAndCompute(total_params, (3.0 * total_compute))
        self.total_conv1d_compute = (total_conv1d_compute * 3.0)
        return self.p_and_c_mamba2

    def _att_compute(self):
        attention = ((((4.0 * self.S) * self.head_dim) * self.head_num) * self.layer_types.count('*'))
        softmax = (((3.0 * self.S) * self.head_num) * self.layer_types.count('*'))
        return (attention, softmax)

    def _qkvo_compute(self):
        return ((((2.0 * self.d_model) * (self.group + self.head_num)) * self.head_dim) * self.layer_types.count('*'))

    def count_ffn(self):
        para = (((3.0 * self.d_model) * self.d_ffn) * self.layer_types.count('-'))
        self.p_and_c_ffn = ParaAndCompute(para, (6.0 * para))
        return self.p_and_c_ffn

    def count_other(self):
        para = (self.d_model * self.L)
        self.p_and_c_other = ParaAndCompute(para, ((7.5 * 3.0) * para))
        return self.p_and_c_other

    def parameter_count(self):
        self.p_and_c_non_vocab += self.count_attention()
        self.p_and_c_non_vocab += self.count_ffn()
        self.p_and_c_non_vocab += self.count_other()
        self.p_and_c_non_vocab += self.count_mamba2()
        self.p_and_c_non_vocab.to_float()
        self.para_and_compute = ((self.para_and_compute + self.p_and_c_non_vocab) + self.count_vocab())
        self.para_and_compute.to_float()
        return self.para_and_compute.to_float()

def search_hyper_ssm_atch(para, emb_ratio, wo_emb_ratio, bound1, bound2, tight_bound, with_emb=True, alpha_low=1.0, alpha_high=100.0, s_fix=2048, head_dim=128, print_out=True, hybrid_attention_ratio=0.2, hybrid_mlp_ratio=0.4, consider_compute=False, error_bound=0.005):
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
                    head_dim = (d // head_num)
                    if (head_dim not in [64, 128, 192, 256]):
                        continue
                    d = (head_dim * head_num)
                    dff = (d * ffn_expand)
                    model = HyperMamba(s_len=s_fix, dim_model=d, dim_dff=dff, layer=layer, head_dim=head_dim, v_dim=v_fix, hybrid_attention_ratio=hybrid_attention_ratio, hybrid_mlp_ratio=hybrid_mlp_ratio)
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
