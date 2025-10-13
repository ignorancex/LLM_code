
import matplotlib.pyplot as plt
from scalinglaw_utils.architecture_search.search_arh_baseLLM import search_ach_baseLLM
from scalinglaw_utils.architecture_search.llm_architecture_base import *
from scalinglaw_utils.architecture_search.llm_architecture_dense_mfa import MLAArchitecture, MFAArchitecture1, MFAArchitecture2, MFAArchitecture3, MFAArchitecture4
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '--']

class MFAAnalysis(object):

    @staticmethod
    def cal_para_computer_per_layer(D, S, n, u, with_value=False):
        factor = (2 if with_value else 1)
        N_MFA_1 = ((((2 * u) + ((2 * u) / n)) + ((factor * (u ** 2)) / n)) * (D ** 2))
        M_MFA_1 = ((((4 * u) * D) * S) + (2 * N_MFA_1))
        N_MFA_2 = (((u + ((3 * u) / n)) + ((factor * (u ** 2)) / n)) * (D ** 2))
        M_MFA_2 = ((((4 * u) * D) * S) + (2 * N_MFA_2))
        N_MFA_3 = (((1 + ((4 * u) / n)) + ((factor * (u ** 2)) / n)) * (D ** 2))
        M_MFA_3 = ((((4 * u) * D) * S) + (2 * N_MFA_3))
        N_MLA = ((1 + ((6 * u) / n)) * (D ** 2))
        M_MLA = (((4 * D) * S) + (2 * N_MLA))
        N_MHA = (4 * (D ** 2))
        M_MHA = (((4 * D) * S) + (2 * N_MHA))
        return (N_MFA_1, M_MFA_1, N_MFA_2, M_MFA_2, N_MFA_3, M_MFA_3, N_MLA, M_MLA, N_MHA, M_MHA)

    @staticmethod
    def cal_para_computer(D, S, n, u, fnn, layer, v_dim, head_dim, with_value=False, ideal_head_num=False):
        model_list = [None, None, None, None, None]
        (res, equal_models) = ([], [])
        model_list[0] = MFAArchitecture1(S, D, fnn, layer, v_dim, head_dim, n, u, with_value, ideal_head_num)
        model_list[1] = MFAArchitecture2(S, D, fnn, layer, v_dim, head_dim, n, u, with_value, ideal_head_num)
        if with_value:
            model_list[2] = MFAArchitecture4(S, D, fnn, layer, v_dim, head_dim, n, u, True, ideal_head_num)
        else:
            model_list[2] = MFAArchitecture3(S, D, fnn, layer, v_dim, head_dim, n, u, True, ideal_head_num)
        model_list[3] = MLAArchitecture(S, D, fnn, layer, v_dim, head_dim, n, u)
        model_list[4] = BaseLLMArchitecture(S, D, fnn, layer, v_dim, head_dim, n)
        for model in model_list:
            model.parameter_count()
            res.append(model.p_and_c_non_vocab.para)
            res.append(model.p_and_c_non_vocab.compute)
        if ideal_head_num:
            return res
        for i in range(3):
            origin_print = 0
            if (abs(((model_list[i].p_and_c_non_vocab.para / model_list[4].p_and_c_non_vocab.para) - 1)) < 0.3):
                if (abs(((model_list[i].p_and_c_non_vocab.para / model_list[4].p_and_c_non_vocab.para) - 1)) < 0.001):
                    print(('critical model: %s' % model_list[i]))
                    print(('origin: %s' % model_list[4]))
                    origin_print = 1
                new_models = model_list[i].search_head_dim()
                if ((new_models is not None) and (len(new_models) > 0)):
                    for r in new_models:
                        error = ((r.p_and_c_non_vocab.para / model_list[4].p_and_c_non_vocab.para) - 1)
                        if ((abs(error) < 0.001) or ((i == 1) and (abs(error) <= 0.002))):
                            equal_models.append(r)
                            if (origin_print < 1):
                                print(('origin model: %s' % model_list[i]))
                                origin_print = 1
                            print(('Error %.5f and model: %s' % (error, r)))
        res.append(equal_models)
        return res

    @staticmethod
    def complex_per_layer(D, S, n, ffn=None, layer=None, v_dim=None, head_dim: float=None, with_value=False, plot=True, ideal_head_num=False):
        (u_list, p_ratio_1, c_ratio_1, p_ratio_2, c_ratio_2, p_ratio_3, c_ratio_3, p_ratio_4, c_ratio_4, equal_models) = ([], [], [], [], [], [], [], [], [], [])
        front = (head_dim - 1)
        for mu in range(1000, 4000, 1):
            u = (float(mu) / 1000.0)
            this = int((u * head_dim))
            if (this == front):
                continue
            if (ffn is not None):
                (N_MFA_1, M_MFA_1, N_MFA_2, M_MFA_2, N_MFA_3, M_MFA_3, N_MLA, M_MLA, N_MHA, M_MHA, e_models) = MFAAnalysis.cal_para_computer(D, S, n, u, ffn, layer, v_dim, head_dim, with_value, ideal_head_num)
                equal_models = MFAAnalysis.merge_two_model_list(equal_models, e_models)
            else:
                (N_MFA_1, M_MFA_1, N_MFA_2, M_MFA_2, N_MFA_3, M_MFA_3, N_MLA, M_MLA, N_MHA, M_MHA) = MFAAnalysis.cal_para_computer_per_layer(D, S, n, u, with_value)
            u_list.append(u)
            p_ratio_1.append((N_MFA_1 / N_MHA))
            c_ratio_1.append((M_MFA_1 / M_MHA))
            p_ratio_2.append((N_MFA_2 / N_MHA))
            c_ratio_2.append((M_MFA_2 / M_MHA))
            p_ratio_3.append((N_MFA_3 / N_MHA))
            c_ratio_3.append((M_MFA_3 / M_MHA))
            p_ratio_4.append((N_MLA / N_MHA))
            c_ratio_4.append((M_MLA / M_MHA))
            front = this
        if plot:
            plt.figure(dpi=500)
            plt.plot(u_list, p_ratio_1, COLORS[0], label=('MFA1%spara_ratio' % ('+v_' if with_value else '_')))
            plt.plot(u_list, c_ratio_1, COLORS[1], label=('MFA1%scompute_ratio' % ('+v_' if with_value else '_')))
            plt.plot(u_list, p_ratio_2, COLORS[2], label=('MFA2%spara_ratio' % ('+v_' if with_value else '_')))
            plt.plot(u_list, c_ratio_2, COLORS[3], label=('MFA2%scompute_ratio' % ('+v_' if with_value else '_')))
            plt.plot(u_list, p_ratio_3, COLORS[4], label=('MFA3%spara_ratio' % ('+v_' if with_value else '_')))
            plt.plot(u_list, c_ratio_3, COLORS[5], label=('MFA3%scompute_ratio' % ('+v_' if with_value else '_')))
            plt.plot(u_list, p_ratio_4, COLORS[6], label='MLA_para_ratio')
            plt.plot(u_list, c_ratio_4, COLORS[7], label='MLA_compute_ratio')
            plt.xlabel('mu: C/d')
            plt.ylabel('ratio: %')
            title = ('D=%d_S=%d_n=%d%s%s' % (D, S, n, ('' if (layer is None) else ('_L=%d' % layer)), ('' if (ffn is None) else ('_F=%d' % ffn))))
            plt.title(title)
            plt.legend()
            plt.grid()
            plt.savefig((('MFA_fig/' + title) + '.png'))
            plt.close()
        return equal_models

    @staticmethod
    def dedup_models(models, res=None):
        if (res is None):
            res = list()
        for model in models:
            count = 0
            for r in res:
                if (model != r):
                    count += 1
            if (count == len(res)):
                res.append(model)
        return res

    @staticmethod
    def merge_two_model_list(dedup, models):
        if ((models is None) or (len(models) == 0)):
            return dedup
        d_models = MFAAnalysis.dedup_models(models)
        return MFAAnalysis.dedup_models(d_models, dedup)

def MFA_analysis():
    models = search_ach_baseLLM((1073810000.0 / 2.5), 5.95, 6.75, 0.1, 0.1, 0.03, False, 3.5, 10, 2048, 128, False)
    for model in models:
        res1 = MFAAnalysis.complex_per_layer(model.d_model, model.S, model.head_num, model.d_ffn, model.L, model.V, model.head_dim, False, True, False)
        res2 = MFAAnalysis.complex_per_layer(model.d_model, model.S, model.head_num, model.d_ffn, model.L, model.V, model.head_dim, True, True, False)
        print(('Searched origin model is:\n%s' % model))
        print('Equal model are:')
        for r in MFAAnalysis.merge_two_model_list(res1, res2):
            print(r)

def print_1b_std():
    (S, D, fnn, layer, v_dim, head_dim, n, mu) = (2048, 2048, 6008, 20, 65536, 128, 16, 1)
    print(BaseLLMArchitecture(S, D, fnn, layer, v_dim, head_dim, n).p_count())
    print(GQAArchitecture(S, D, fnn, layer, v_dim, (head_dim / 2), (n * 2), 8).p_count())
    print(GQAArchitecture(S, D, fnn, layer, v_dim, head_dim, n, 4).p_count())
    print(GQAArchitecture(S, D, fnn, layer, v_dim, head_dim, n, 1).p_count())
    print(MFAArchitecture1(S, D, fnn, layer, v_dim, head_dim, n, mu, False).p_count())
    print(MFAArchitecture1(S, D, fnn, layer, v_dim, head_dim, n, mu, True).p_count())
    print(MFAArchitecture2(S, D, fnn, layer, v_dim, head_dim, n, mu, False).p_count())
    print(MFAArchitecture2(S, D, fnn, layer, v_dim, (head_dim * 2), (n / 2), mu, False).p_count())
    print(MFAArchitecture2(S, D, fnn, layer, v_dim, head_dim, n, mu, True).p_count())
    print(MFAArchitecture2(S, D, fnn, layer, v_dim, (head_dim * 2), (n / 2), mu, True).p_count())
    print(MLAArchitecture(S, D, fnn, layer, v_dim, head_dim, n, mu).p_count())
    print(MLAArchitecture(S, D, fnn, layer, v_dim, (head_dim * 2), (n / 2), mu).p_count())
if (__name__ == '__main__'):
    print_1b_std()
