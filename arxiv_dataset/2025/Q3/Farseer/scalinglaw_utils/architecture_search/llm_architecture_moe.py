
from scalinglaw_utils.architecture_search.llm_architecture_base import BaseLLMArchitecture, ParaAndCompute

class BaseMoEArchitecture(BaseLLMArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, expert_num=0, top_k=0, dim_dynamic_ffn=0, dim_share_ffn=0, moe_layer=0):
        super().__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num)
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

    def _base_str(self):
        base_str = super()._base_str()
        return (base_str + ('E:%d, K:%d, De:%d, Ds:%d, Le:%d, ' % (self.E, self.K, self.de, self.ds, self.Le)))

    def _ratio_str(self):
        return ('sparsity:%.4f, beta:%.2f, mu:%.2f, \natt_ratio:%.2f, gamma:%.2f, rho:%.2f, zeta:%.1f' % (self.sparsity, self.beta, self.mu, self.att_ratio, self.gamma, self.rho, self.zeta))

    def __str__(self):
        return ((((((((self._base_str() + self._att_str()) + self._ratio_str()) + ', non_emb para:') + str(self.p_and_c_non_vocab)) + ', act para:') + str(self.p_and_c_act)) + ', para:') + str(self.para_and_compute))

    def __eq__(self, other):
        res = super().__eq__(other)
        res = (res and (self.E == other.E) and (self.K == other.K) and (self.ds == other.ds) and (self.de == other.de) and (self.Le == other.Le))
        return res

    def __hash__(self):
        return hash((self.S, self.d_model, self.d_ffn, self.L, self.V, self.head_dim, self.head_num, self.E, self.K, self.ds, self.de, self.Le))

    def short_msg(self):
        return (((('D:%d, Df:%d, L:%d, E:%d, K:%d, De:%d, Ds:%d, Le:%d, tao:%.4f, beta:%.2f, mu:%.2f, zeta:%.1f, N:' % (self.d_model, self.d_ffn, self.L, self.E, self.K, self.de, self.ds, self.Le, self.sparsity, self.beta, self.mu, self.zeta)) + '{:.3e}'.format(self.p_and_c_non_vocab.para)) + ', NA:') + '{:.3e}'.format(self.p_and_c_act.para))

    def parameter_count(self):
        super().parameter_count()
        self.p_and_c_act = (((self.p_and_c_act + self.p_and_c_ffn_act) + self.p_and_c_other) + self.p_and_c_att)
        self.sparsity = (self.p_and_c_act.para / self.p_and_c_non_vocab.para)
        self.att_ratio = (self.p_and_c_att.para / self.p_and_c_non_vocab.para)
        return self.para_and_compute.to_float()

    def count_ffn(self):
        para = (((3.0 * self.d_model) * self.d_ffn) * (self.L - self.Le))
        para_router = ((self.d_model * self.E) * self.Le)
        para_share = (((3.0 * self.d_model) * self.ds) * self.Le)
        para_moe_fnn = ((((3.0 * self.d_model) * self.de) * self.E) * self.Le)
        para_moe_act = ((((3.0 * self.d_model) * self.de) * self.K) * self.Le)
        para_all = (((para + para_share) + para_moe_fnn) + para_router)
        para_act = (((para + para_share) + para_moe_act) + para_router)
        self.p_and_c_ffn = ParaAndCompute(para_all, (6.0 * para_act))
        self.p_and_c_ffn_act = ParaAndCompute(para_act, (6.0 * para_act))
        return self.p_and_c_ffn

    def p_count(self):
        self.p_and_c_act.reset()
        return super().p_count()

    def reset_layer(self, L):
        self.Le = (L - (self.L - self.Le))
        return super().reset_layer(L)

def use_case_36B():
    moe_36 = BaseMoEArchitecture(16384, 3584, 9600, 40, 65536, 128, None, 32, 2, 2368, 4736, 39)
    print(moe_36.p_count())

def use_case_2B():
    model1 = BaseMoEArchitecture(16384, 1408, 3904, 17, 65536, 64, (1408 / 64), 35, 2, 800, 1600, 16)
    model2 = BaseMoEArchitecture(16384, 1408, 3904, 17, 65536, 64, (1408 / 64), 37, 4, 800, 0, 16)
    model3 = BaseMoEArchitecture(16384, 1408, 3904, 16, 65536, 64, (1408 / 64), 68, 2, 800, 1600, 8)
    model4 = BaseMoEArchitecture(16384, 1408, 3904, 16, 65536, 64, (1408 / 64), 70, 4, 800, 0, 8)
    print(model1.p_count())
    print(model2.p_count())
    print(model3.p_count())
    print(model4.p_count())
if (__name__ == '__main__'):
    use_case_2B()
