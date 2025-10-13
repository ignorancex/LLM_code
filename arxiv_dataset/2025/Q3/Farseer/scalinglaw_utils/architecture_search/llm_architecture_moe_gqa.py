
from scalinglaw_utils.architecture_search.llm_architecture_moe import BaseMoEArchitecture
from scalinglaw_utils.architecture_search.llm_architecture_moe_with_mfa import MoEMFAArchitecture

class MoEGQAArchitecture(BaseMoEArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, expert_num=0, top_k=0, dim_dynamic_ffn=0, dim_share_ffn=0, moe_layer=0, group=8):
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num, expert_num, top_k, dim_dynamic_ffn, dim_share_ffn, moe_layer)
        self.group = float(group)

    def short_msg(self):
        return (((('D:%d, Df:%d, L:%d, E:%d, K:%d, De:%d, Ds:%d, Le:%d, d:%d, hn:%d, tao:%.4f, beta:%.2f, mu:%.2f, zeta:%.1f, N:' % (self.d_model, self.d_ffn, self.L, self.E, self.K, self.de, self.ds, self.Le, self.head_dim, self.head_dim, self.sparsity, self.beta, self.zeta, self.mu)) + '{:.3e}'.format(self.p_and_c_non_vocab.para)) + ', NA:') + '{:.3e}'.format(self.p_and_c_act.para))

    def _qkvo_compute(self):
        wqo = ((((2 * self.d_model) * self.head_num) * self.head_dim) * self.L)
        wkv = ((((2 * self.d_model) * self.group) * self.head_dim) * self.L)
        return (wkv + wqo)

    def get_kvcache_size(self, sequence_length: float):
        return ((((sequence_length * self.L) * self.head_dim) * self.group) * 2.0)

def use_case_36B():
    moe_36_mha = BaseMoEArchitecture(16384, 3584, 9600, 40, 65536, 128, None, 32, 3, 2368, 4352, 39)
    moe_36_gqa = MoEGQAArchitecture(16384, 3584, 9600, 40, 65536, 128, None, 16, 2, ((2368 * 2) + 128), (64 * 55), 39)
    moe_36_mfa = MoEMFAArchitecture(16384, 3584, 9600, 40, 65536, 128, 40, 16, 2, 4864, 4736, 39)
    moe_10_mfa = MoEMFAArchitecture(16384, 3584, 9600, 10, 65536, 128, 40, 16, 2, 4864, 4736, 9)
    [print(model.p_count()) for model in [moe_36_mha, moe_36_gqa, moe_36_mfa, moe_10_mfa]]
    print(moe_36_mfa.reset_layer(10))
if (__name__ == '__main__'):
    use_case_36B()
