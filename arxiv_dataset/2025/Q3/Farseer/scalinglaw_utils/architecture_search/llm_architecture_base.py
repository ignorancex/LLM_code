
STD_HEAD_DIM_SETTING = [32, 64, 128, 192, 256]
DATA_CLASS_TO_BYTE_NUM = {'fp32': 4.0, 'bf16': 2.0, 'fp16': 2.0, 'int8': 1.0, 'fp8': 1.0}

class ParaAndCompute():

    def __init__(self, para, compute):
        self.para = para
        self.compute = compute

    def __add__(self, other):
        self.para += other.para
        self.compute += other.compute
        return self

    def __sub__(self, other):
        self.para -= other.para
        self.compute -= other.compute
        return self

    def __str__(self):
        return (('{:.5e}'.format(self.para) + ',') + '{:.5e}'.format(self.compute))

    def __copy__(self):
        return ParaAndCompute(self.para, self.compute)

    def __eq__(self, other):
        return ((self.para == other.para) and (self.compute == other.compute))

    def to_float(self):
        self.para = float(self.para)
        self.compute = float(self.compute)
        return self

    def compute_para_ratio(self):
        return (float(self.compute) / (self.para + 1e-10))

    def reset(self):
        self.para = 0.0
        self.compute = 0.0

class BaseLLMArchitecture():

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None):
        self.S = float(s_len)
        self.d_model = float(dim_model)
        self.d_ffn = float(dim_dff)
        self.L = float(layer)
        self.V = float(v_dim)
        self.head_dim = float(head_dim)
        self.head_num = float(((self.d_model // self.head_dim) if (head_num is None) else head_num))
        self.alpha = (dim_dff / dim_model)
        self.gamma = (dim_model / layer)
        self.para_and_compute = ParaAndCompute(0, 0)
        self.p_and_c_non_vocab = ParaAndCompute(0, 0)

    def _base_str(self):
        return ('S:%d, D:%d, ffn:%d, L:%d, V:%d, ' % (self.S, self.d_model, self.d_ffn, self.L, self.V))

    def _att_str(self):
        return ('head_dim:%d, head:%d, ' % (self.head_dim, self.head_num))

    def _ratio_str(self):
        return ('alpha:%.2f, gamma:%.2f, att_ratio:%.2f, p2c_ratio:%.3f, p2c_ratio_wo_emb:%.3f,' % (self.alpha, self.gamma, (self.p_and_c_att.para / self.p_and_c_non_vocab.para), self.para_and_compute.compute_para_ratio(), self.p_and_c_non_vocab.compute_para_ratio()))

    def __str__(self):
        return ((((((self._base_str() + self._att_str()) + self._ratio_str()) + '  para:') + str(self.para_and_compute)) + ', non_emb para:') + str(self.p_and_c_non_vocab))

    def __eq__(self, other):
        res = (self.__class__ == other.__class__)
        res = (res and (self.S == other.S) and (self.d_model == other.d_model) and (self.d_ffn == other.d_ffn))
        res = (res and (self.L == other.L) and (self.V == other.V))
        res = (res and (self.head_dim == other.head_dim) and (self.head_num == other.head_num))
        return res

    def __hash__(self):
        return hash((self.S, self.d_model, self.d_ffn, self.L, self.V, self.head_dim, self.head_num))

    def parameter_count(self):
        self.p_and_c_non_vocab += self.count_attention()
        self.p_and_c_non_vocab += self.count_ffn()
        self.p_and_c_non_vocab += self.count_other()
        self.p_and_c_non_vocab.to_float()
        self.para_and_compute = ((self.para_and_compute + self.p_and_c_non_vocab) + self.count_vocab())
        return self.para_and_compute.to_float()

    def p_count(self):
        self.p_and_c_non_vocab.reset()
        self.para_and_compute.reset()
        self.parameter_count()
        return self

    def count_vocab(self):
        para = ((2.0 * self.d_model) * self.V)
        self.p_and_c_voc = ParaAndCompute(para, (3.0 * para))
        return self.p_and_c_voc

    def _att_compute(self):
        attention = ((((2.0 * self.S) * self.head_dim) * self.head_num) * self.L)
        softmax = (((3.0 * self.S) * self.head_num) * self.L)
        return (attention, softmax)

    def _qkvo_compute(self):
        return ((((4.0 * self.d_model) * self.head_num) * self.head_dim) * self.L)

    def count_attention(self):
        qkvo = self._qkvo_compute()
        (attention, softmax) = self._att_compute()
        self.p_and_c_att = ParaAndCompute(qkvo, (3.0 * (((2.0 * qkvo) + attention) + softmax)))
        return self.p_and_c_att

    def count_ffn(self):
        para = (((3.0 * self.d_model) * self.d_ffn) * self.L)
        self.p_and_c_ffn = ParaAndCompute(para, (6.0 * para))
        return self.p_and_c_ffn

    def count_other(self):
        para = ((2.0 * self.d_model) * self.L)
        self.p_and_c_other = ParaAndCompute(para, ((7.5 * 3.0) * para))
        return self.p_and_c_other

    def get_kvcache_size(self, sequence_length: float):
        return ((((sequence_length * self.L) * self.head_dim) * self.head_num) * 2.0)

    def get_inference_mem_size(self, sequence_length: float=None, att_type: str='fp16', other_type: str='int8', kv_cache_type: str='fp16'):
        if (self.p_and_c_non_vocab.para == 0):
            self.parameter_count()
        sequence_length = (self.S if ((sequence_length is None) or (sequence_length < 0)) else float(sequence_length))
        fix_mem = (((self.para_and_compute.para - self.p_and_c_att.para) * DATA_CLASS_TO_BYTE_NUM[other_type.lower()]) + (self.p_and_c_att.para * DATA_CLASS_TO_BYTE_NUM[att_type.lower()]))
        kv_cache_size = (self.get_kvcache_size(sequence_length) * DATA_CLASS_TO_BYTE_NUM[kv_cache_type.lower()])
        return ((fix_mem + kv_cache_size), fix_mem)

    def get_train_mem_size(self, sequence_length: float=None, opt_type='adam', base_time=6):
        opt_factor = (16 if (opt_type == 'adam') else 12)
        if (self.p_and_c_non_vocab.para == 0):
            self.parameter_count()
        sequence_length = (self.S if ((sequence_length is None) or (sequence_length < 0)) else float(sequence_length))
        ffn_time = (self.alpha if (not hasattr(self, 'beta')) else getattr(self, 'beta'))
        fix_mem = (self.para_and_compute.para * opt_factor)
        act_mem = ((((2 * self.L) * self.d_model) * (base_time + (2 * ffn_time))) * sequence_length)
        return ((fix_mem + act_mem), fix_mem)

    def get_max_seq_len_with_memory(self, max_mem: float, att_type: str='fp16', other_type: str='int8', kv_cache_type: str='fp16'):
        (all_mem, fix_mem) = self.get_inference_mem_size(1, att_type, other_type, kv_cache_type)
        return ((float(max_mem) - fix_mem) / (all_mem - fix_mem))

    def reset_layer(self, L):
        self.L = float(L)
        return self.p_count()

class GQAArchitecture(BaseLLMArchitecture):

    def __init__(self, s_len, dim_model, dim_dff, layer, v_dim, head_dim=128, head_num=None, group=8):
        super(GQAArchitecture, self).__init__(s_len, dim_model, dim_dff, layer, v_dim, head_dim, head_num)
        self.group = float(group)
        assert ((self.head_num % self.group) == 0)

    def _att_str(self):
        return ('head_dim:%d, head:%d, group:%d, ' % (self.head_dim, self.head_num, self.group))

    def _qkvo_compute(self):
        return ((((2.0 * self.d_model) * (self.group + self.head_num)) * self.head_dim) * self.L)

def use_case_count_llama():
    llama_2_7_b = BaseLLMArchitecture(4096, 4096, 11008, 32, 32000)
    llama_1_65_b = BaseLLMArchitecture(2048, 8192, 22016, 80, 32000)
    qwen_72_b = BaseLLMArchitecture(2048, 8192, 24576, 80, 151936)
    print(llama_2_7_b.parameter_count())
    print(llama_2_7_b)
    print(llama_1_65_b.parameter_count())
    print(qwen_72_b.parameter_count())

def use_case_count_qwen():
    qwen_7b = GQAArchitecture(8192, 3584, (18944 / 1.5), 28, 152064, 128, 28, 4)
    print('===== Qwen 7B Wrong =========')
    print(qwen_7b.p_count())
    qwen_7b = GQAArchitecture(8192, 3584, 18944, 28, 152064, 128, 28, 4)
    print('===== Qwen 7B Correct =========')
    print(qwen_7b.p_count())
    samples = 1000000
    token_num = ((8192 * 16) * samples)
    all_length = (8192 * 16)
    chunk_sizes = [512, 1024, 2048, 4096, 8192, (8192 * 2), (8192 * 4), (8192 * 8)]
    for chunk_size in chunk_sizes:
        chunk_num = (all_length / chunk_size)
        qwen_7b = GQAArchitecture((chunk_size * 2), 3584, 18944, 28, 152064, 128, 28, 4)
        per_token_compute = (qwen_7b.p_count().para_and_compute.compute / 3)
        print(('=====  for chunk size %d, samples num %d =========' % (chunk_size, samples)))
        total_compute = ((((per_token_compute * chunk_size) * chunk_num) * (chunk_num - 1)) * samples)
        print(total_compute)
        print(((((total_compute * 2) / 332000000000000.0) / 3600) / 24))
if (__name__ == '__main__'):
    model = BaseLLMArchitecture(8192, 2304, 6064, 36, 32000)
    print(model.p_count())
    print((((((model.para_and_compute.compute * 1024) * 8192) / 4) / 989000000000000) / 128))
