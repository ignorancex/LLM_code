
from scalinglaw_utils.architecture_search.llm_architecture_base import BaseLLMArchitecture, STD_HEAD_DIM_SETTING, GQAArchitecture

class MLAArchitecture(BaseLLMArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, multiply=1):
        super(MLAArchitecture, self).__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num)
        self.multiply = float(multiply)
        self.low_rank_dim = float(int((self.multiply * self.head_dim)))

    def _att_str(self):
        return ('head_dim:%d, head:%d, low_rank:%d, ' % (self.head_dim, self.head_num, self.low_rank_dim))

    def _qkvo_compute(self):
        wo = (((self.d_model * self.head_num) * self.head_dim) * self.L)
        qkv = (((6.0 * self.d_model) * self.low_rank_dim) * self.L)
        return (wo + qkv)

class MFAArchitecture1(BaseLLMArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128.0, head_num=None, multiply: float=1.0, multi_value=True, ideal_multi_head=False):
        super(MFAArchitecture1, self).__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num)
        self.multiply = float(multiply)
        self.low_rank_dim = (float(int((self.multiply * float(self.head_dim)))) if (not ideal_multi_head) else self.head_dim)
        self.head_num = ((head_num * multiply) if ideal_multi_head else head_num)
        self.multi_value = multi_value

    def __eq__(self, other):
        res = super(MFAArchitecture1, self).__eq__(other)
        return (res and (self.multi_value == other.multi_value) and (self.low_rank_dim == other.low_rank_dim))

    def _att_class(self):
        return 'MFA1'

    def _att_str(self):
        s = ('%s, head_dim:%d, head:%d, low_rank:%d' % (self._att_class(), self.head_dim, self.head_num, self.low_rank_dim))
        return ((s + '+v, ') if self.multi_value else (s + ', '))

    def _qkvo_compute(self):
        wqo = ((((2.0 * self.d_model) * self.head_num) * self.low_rank_dim) * self.L)
        wkv = ((((2.0 * self.d_model) * self.low_rank_dim) * self.L) + ((self.head_num * (self.low_rank_dim ** 2)) * self.L))
        if self.multi_value:
            wkv += ((self.head_num * (self.low_rank_dim ** 2)) * self.L)
        return (wqo + wkv)

    def _att_compute(self):
        attention = ((((2.0 * self.S) * self.low_rank_dim) * self.head_num) * self.L)
        softmax = (((3.0 * self.S) * self.head_num) * self.L)
        return (attention, softmax)

    def search_head_dim(self):
        para = self.p_and_c_non_vocab.to_float().para
        att_all_size = (self.low_rank_dim * self.head_num)
        res = []
        for i in range(2, 5):
            dim = STD_HEAD_DIM_SETTING[i]
            head = (att_all_size // dim)
            new_model = self.__new__(self.__class__)
            new_model.__init__(self.S, self.d_model, self.d_ffn, self.L, self.V, self.head_dim, head, (float(dim) / float(self.head_dim)), self.multi_value)
            new_model.parameter_count()
            new_para = new_model.p_and_c_non_vocab.to_float().para
            if ((abs((new_para - para)) / para) < 0.1):
                res.append(new_model)
        return res

class MFAArchitecture2(MFAArchitecture1):

    def _att_class(self):
        return 'MFA2'

    def _qkvo_compute(self):
        wo = (((self.d_model * self.head_num) * self.low_rank_dim) * self.L)
        qkv_share = (((self.d_model * 3) * self.low_rank_dim) * self.L)
        wq = ((self.head_num * (self.low_rank_dim ** 2)) * self.L)
        wv = (wq if self.multi_value else 0)
        return (((wo + qkv_share) + wq) + wv)

class MFAArchitecture3(MFAArchitecture1):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, multiply=1, multi_value=True, ideal_head_num=False):
        super(MFAArchitecture3, self).__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num, multiply, multi_value, ideal_head_num)
        assert multi_value

    def _att_class(self):
        return 'MFA3'

    def _qkvo_compute(self):
        qkv_share = (((self.d_model * 3.0) * self.low_rank_dim) * self.L)
        wo = (((self.d_model * self.head_num) * self.head_dim) * self.L)
        wq = ((self.head_num * (self.low_rank_dim ** 2)) * self.L)
        wv = ((((self.head_num * self.low_rank_dim) * self.head_dim) * self.L) if self.multi_value else 0)
        return (((wo + qkv_share) + wq) + wv)

class MFAArchitecture4(MFAArchitecture3):

    def _att_class(self):
        return 'MFA4'

    def _qkvo_compute(self):
        return (super(MFAArchitecture4, self)._qkvo_compute() + ((self.head_num * (self.low_rank_dim ** 2)) * self.L))

def use_case():
    args = [2048, 2048, 6008, 20, 65536, 128, 16]
    mha1B = BaseLLMArchitecture(*args)
    mqa1B = GQAArchitecture(*args, group=1)
    gqa1B = GQAArchitecture(*args, group=4)
    mfa1 = MFAArchitecture1(*args, multi_value=False)
    mfa2a = MFAArchitecture2(*args, multi_value=False)
    mfa2b = MFAArchitecture2(2048, 2048, 6008, 20, 65536, 256, 8, multi_value=False)
    mla = MLAArchitecture(*args)
    for model in [mha1B, mqa1B, gqa1B, mfa1, mfa2a, mfa2b, mla]:
        print(model.p_count())
if (__name__ == '__main__'):
    use_case()
