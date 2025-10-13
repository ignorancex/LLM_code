
from scalinglaw_utils.architecture_search.llm_architecture_moe import BaseMoEArchitecture

class MoEMFAArchitecture(BaseMoEArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, expert_num=0, top_k=0, dim_dynamic_ffn=0, dim_share_ffn=0, moe_layer=0, share_q_dim=None):
        if (share_q_dim is None):
            self.share_q_dim = head_dim
        else:
            self.share_q_dim = share_q_dim
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num, expert_num, top_k, dim_dynamic_ffn, dim_share_ffn, moe_layer)

    def short_msg(self):
        sparsity = (self.p_and_c_act.para / self.p_and_c_non_vocab.para)
        mfa_ratio = (((((((2 * self.d_model) * self.head_dim) + (self.d_model * self.share_q_dim)) + (((self.d_model + self.share_q_dim) * self.head_num) * self.head_dim)) / self.d_model) / self.d_model) / 4)
        return (((('D:%d, Df:%d, L:%d, E:%d, K:%d, De:%d, Ds:%d, Le:%d, hd:%d, hn:%d, AR:%.4f, beta:%.2f, mu:%.2f, zeta:%.1f, mfa:%.3f, N:' % (self.d_model, self.d_ffn, self.L, self.E, self.K, self.de, self.ds, self.Le, self.head_dim, self.head_num, sparsity, self.beta, self.mu, self.zeta, mfa_ratio)) + '{:.3e}'.format(self.p_and_c_non_vocab.para)) + ', NA:') + '{:.3e}'.format(self.p_and_c_act.para))

    def _att_compute(self):
        attention = ((((2.0 * self.S) * self.head_dim) * self.head_num) * self.L)
        softmax = (((3.0 * self.S) * self.head_num) * self.L)
        return (attention, softmax)

    def _qkvo_compute(self):
        wo = (((self.d_model * self.head_num) * self.head_dim) * self.L)
        qkv_share = ((((2 * self.d_model) * self.head_dim) + (self.d_model * self.share_q_dim)) * self.L)
        wq = ((self.head_num * (self.head_dim * self.share_q_dim)) * self.L)
        return ((wo + qkv_share) + wq)

    def get_kvcache_size(self, sequence_length: float):
        return (((sequence_length * self.L) * self.head_dim) * 2.0)

def use_case_7B():
    moe_7 = MoEMFAArchitecture(8192, 2048, 6144, 24, 128815, 256, 18, 29, 2, 1504, 3008, 23)
    print(moe_7.p_count())
    print((((((moe_7.para_and_compute.compute * 512) * 8192) / 1.85) / 989000000000000) / 128))

def use_case_36B():
    moe_36 = BaseMoEArchitecture(16384, 3584, 9600, 40, 65536, 128, None, 32, 3, 2368, ((4736 - 256) - 128), 39)
    print(moe_36.p_count())
    moe_34 = MoEMFAArchitecture(16384, 3584, 9600, 40, 65536, 128, (3584 / 128), 32, 2, 2368, 4736, 39)
    print(moe_34.p_count())
    moe_34 = MoEMFAArchitecture(16384, 3584, 9600, 41, 65536, 128, (3584 / 128), 32, 2, 2368, 4736, 40)
    print(moe_34.p_count())
    moe_36 = MoEMFAArchitecture(16384, 3584, 9600, 40, 65536, 128, ((3584 / 128) + 12), 16, 2, ((2368 * 2) + 128), (2368 * 2), 39)
    print(moe_36.p_count())

def use_case_36B_infer_mem():
    moe_36 = MoEMFAArchitecture(16384, 3584, 9600, 40, 65536, 128, ((3584 / 128) + 12), 16, 2, ((2368 * 2) + 128), (2368 * 2), 39)
    print(moe_36.get_inference_mem_size(0))
    moe_36_mha = BaseMoEArchitecture(16384, 3584, 9600, 40, 65536, 128, None, 32, 3, 2368, ((4736 - 256) - 128), 39)
    print(moe_36_mha.get_inference_mem_size(0))

def use_case_300B_mem():
    moe_300_mfa = MoEMFAArchitecture(8192, 7168, 18688, 65, 129280, 256, 80, 32, 2, 7168, 7168, 62)
    print(moe_300_mfa.p_count())
    print(moe_300_mfa.short_msg())
if (__name__ == '__main__'):
    use_case_7B()
