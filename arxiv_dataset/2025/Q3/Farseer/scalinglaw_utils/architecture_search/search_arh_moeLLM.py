
import math
import time
from scalinglaw_utils.architecture_search.llm_architecture_moe import BaseMoEArchitecture
from scalinglaw_utils.architecture_search.llm_architecture_moe_with_mfa import MoEMFAArchitecture
PARA_LOW_BOUND = 0.95
PARA_HIGH_BOUND = 1.05
FFN_DIM_MULTIPLIER = 32
MAX_EXPERT_NUM = 100
PAPA_ERROR_BOUND = 0.01

def find_nearest_divisible(target, divisor):
    upper = (math.ceil((target / divisor)) * divisor)
    lower = (math.floor((target / divisor)) * divisor)
    if (abs((upper - target)) < abs((lower - target))):
        return upper
    else:
        return lower

def search_arh_moeLLM(N: float, NA: float, beta_low: float, beta_up: float, l_low: float=6, l_up: float=128, alpha: float=3, ld: float=1, head_dim: float=128, rho: float=0.5, s_fix: float=2048, v_dim=65536):
    model_list = dict()
    (N_low, N_up, NA_low, NA_up) = ((PARA_LOW_BOUND * N), (PARA_HIGH_BOUND * N), (PARA_LOW_BOUND * NA), (PARA_HIGH_BOUND * NA))
    dense_factor = ((4.0 + (3.0 * alpha)) * ld)
    for layer in range(int(l_low), (int(l_up) + 1)):
        layer = float(layer)
        moe_factor_low = ((4.0 + (3.0 * beta_low)) * layer)
        moe_factor_up = ((4.0 + (3.0 * beta_up)) * layer)
        model_dim_low = math.sqrt((NA / (moe_factor_up + dense_factor)))
        model_dim_up = math.sqrt((NA / (moe_factor_low + dense_factor)))
        head_num_low = math.ceil((model_dim_low / head_dim))
        head_num_up = math.floor((model_dim_up / head_dim))
        for head_num in range(int(head_num_low), (int(head_num_up) + 1)):
            model_dim = float((head_num * head_dim))
            dense_para = (dense_factor * (model_dim ** 2))
            att_para = ((4 * layer) * (model_dim ** 2))
            fnn_fac = ((3 * model_dim) * layer)
            (moe_fnn_up, moe_fnn_low) = ((((N_up - dense_para) - att_para) / fnn_fac), (((N_low - dense_para) - att_para) / fnn_fac))
            (act_fnn_up, act_fnn_low) = ((((NA_up - dense_para) - att_para) / fnn_fac), (((NA_low - dense_para) - att_para) / fnn_fac))
            for k in range(1, 33):
                top_k = float(k)
                (error, min_error, res_model) = (1, PAPA_ERROR_BOUND, None)
                de_times_low = math.ceil((((act_fnn_low * (1 - rho)) / top_k) / FFN_DIM_MULTIPLIER))
                de_times_up = math.floor((((act_fnn_up * (1 - rho)) / top_k) / FFN_DIM_MULTIPLIER))
                for de_times in range(de_times_low, (de_times_up + 1)):
                    de = (float(de_times) * float(FFN_DIM_MULTIPLIER))
                    share_dim = (((rho / (1 - rho)) * top_k) * de)
                    expert_num_up = min(math.floor(((moe_fnn_up - share_dim) / de)), MAX_EXPERT_NUM)
                    expert_num_low = math.ceil(((moe_fnn_low - share_dim) / de))
                    for expert_num in range(expert_num_low, (expert_num_up + 1)):
                        model = BaseMoEArchitecture(s_fix, model_dim, (model_dim * alpha), (layer + ld), v_dim, head_dim, head_num, expert_num, top_k, de, share_dim, layer)
                        model.parameter_count()
                        error = ((abs((model.p_and_c_non_vocab.para - N)) / N) + (abs((model.p_and_c_act.para - NA)) / NA))
                        if (error < min_error):
                            min_error = error
                            res_model = model
                if (res_model is not None):
                    model_list[res_model] = min_error
    return model_list

def search_arh_moe_mfa(N: float, NA: float, zeta_low: float, zeta_up: float, beta_low: float=0.5, beta_up: float=20, l_low: float=6, l_up: float=128, alpha: float=3, ffn_muti=256, hidden_multi=256, mu_low: float=10, mu_up: float=100, ld: float=1, mfa_low: float=0.5, mfa_high: float=1, error_bound=PAPA_ERROR_BOUND, head_dim: float=128, rho_max: float=0.5, rho_min: float=0.1, s_fix: float=2048, v_dim: float=65536):
    model_list = dict()
    (N_low, N_up, NA_low, NA_up) = ((PARA_LOW_BOUND * N), (PARA_HIGH_BOUND * N), (PARA_LOW_BOUND * NA), (PARA_HIGH_BOUND * NA))
    for layer in range(int(l_low), (int(l_up) + 1)):
        (hidden_low, hidden_high) = ((layer * zeta_low), (layer * zeta_up))
        hidden_dims = [i for i in range((int(hidden_low) + 1), int(hidden_high)) if ((i % hidden_multi) == 0)]
        for hidden_dim in hidden_dims:
            dense_ffn_width = find_nearest_divisible((alpha * hidden_dim), ffn_muti)
            dense_ffn = (((3 * hidden_dim) * dense_ffn_width) * ld)
            (mfa_p_low, mfa_p_high) = (((4 * (hidden_dim ** 2)) * mfa_low), ((4 * (hidden_dim ** 2)) * mfa_high))
            mfa_factor1 = ((3 * hidden_dim) * head_dim)
            mfa_factor2 = ((hidden_dim + head_dim) * head_dim)
            head_nums = [i for i in range((int(((mfa_p_low - mfa_factor1) / mfa_factor2)) + 1), int(((mfa_p_high - mfa_factor1) / mfa_factor2)))]
            for head_num in head_nums:
                mfa_size = (mfa_factor1 + (head_num * mfa_factor2))
                mfa_ratio = ((mfa_size / 4) / (hidden_dim ** 2))
                assert (mfa_low < mfa_ratio < mfa_high)
                mfa_all = (mfa_size * (ld + layer))
                fnn_fac = ((3 * hidden_dim) * layer)
                (moe_fnn_up, moe_fnn_low) = ((((N_up - mfa_all) - dense_ffn) / fnn_fac), (((N_low - mfa_all) - dense_ffn) / fnn_fac))
                (act_fnn_up, act_fnn_low) = ((((NA_up - mfa_all) - dense_ffn) / fnn_fac), (((NA_low - mfa_all) - dense_ffn) / fnn_fac))
                if (((moe_fnn_up / hidden_dim) > mu_up) or ((moe_fnn_low / hidden_dim) < mu_low) or ((act_fnn_up / hidden_dim) > beta_up) or ((act_fnn_low / hidden_dim) < beta_low)):
                    continue
                for k in range(1, 33):
                    top_k = float(k)
                    (error, min_error, res_model) = (1, error_bound, None)
                    de_times_low = math.ceil((((act_fnn_low * (1 - rho_max)) / top_k) / ffn_muti))
                    de_times_up = math.floor((((act_fnn_up * (1 - rho_min)) / top_k) / ffn_muti))
                    for de_times in range(de_times_low, (de_times_up + 1)):
                        de = (float(de_times) * float(ffn_muti))
                        share_dim_min = find_nearest_divisible((((rho_min / (1 - rho_min)) * top_k) * de), ffn_muti)
                        share_dim_max = find_nearest_divisible((((rho_max / (1 - rho_max)) * top_k) * de), ffn_muti)
                        for share_dim in range(share_dim_min, (share_dim_max + ffn_muti), ffn_muti):
                            if (((share_dim + (top_k * de)) > act_fnn_up) and ((share_dim + (top_k * de)) < act_fnn_low)):
                                continue
                            expert_num_up = min(math.floor(((moe_fnn_up - share_dim) / de)), MAX_EXPERT_NUM)
                            expert_num_low = math.ceil(((moe_fnn_low - share_dim) / de))
                            for expert_num in range(expert_num_low, (expert_num_up + 1)):
                                model = MoEMFAArchitecture(s_fix, hidden_dim, dense_ffn_width, (layer + ld), v_dim, head_dim, head_num, expert_num, top_k, de, share_dim, layer)
                                model.parameter_count()
                                error = ((abs((model.p_and_c_non_vocab.para - N)) / N) + (abs((model.p_and_c_act.para - NA)) / NA))
                                if (error < min_error):
                                    min_error = error
                                    res_model = model
                    if (res_model is not None):
                        model_list[res_model] = min_error
    return model_list

def process_models(models: dict, error_bound=0.008, min_topK=1, min_dim=0, sort_by='Layer', max_topK=6, min_zeta=1, max_zeta=300):
    filter_models = list()
    for model in models.keys():
        if ((models[model] < error_bound) and (model.K > min_topK) and (model.d_model > min_dim) and (model.K <= max_topK) and (model.zeta > min_zeta) and (model.zeta <= max_zeta)):
            filter_models.append(model)
    if (sort_by == 'Layer'):
        res = sorted(filter_models, key=(lambda obj: ((obj.L * 10000) + obj.d_model)))
    else:
        res = sorted(filter_models, key=(lambda obj: (obj.L + obj.d_model)))
    previous_layer = 0
    for model in res:
        if (model.L != previous_layer):
            print('\n')
            previous_layer = model.L
        print(model.short_msg(), models[model])
    return res

def search_interleave_moe():
    for layer in range(5, 13):
        models = search_arh_moeLLM(2152453248, 368235648, 2, 3, layer, layer, (3904 / 1408), layer)
        for model in models:
            print(model.short_msg(), models[model])

def search_2B_moe_model():
    models = search_arh_moeLLM(2152453248, 368235648, 2, 3, 10, 20, (3904 / 1408), 0)
    return process_models(models, 0.01, min_dim=896, sort_by='dim')

def search_4B_moe_model():
    for i in range(100):
        times = (1.8 + (float(i) * 0.01))
        models = search_arh_moeLLM((2152453248 * times), (368235648 * times), 2.4, 2.65, 15, 40, (8 / 3), 1)
        process_models(models, 0.02, min_dim=896, sort_by='dim', min_zeta=80, max_zeta=92, min_topK=3)

def search_7B_moe_model():
    for i in range(200):
        times = (3 + (float(i) * 0.005))
        models = search_arh_moeLLM((2152453248 * times), (368235648 * times), 2.4, 2.65, 15, 40, (8 / 3), 1)
        process_models(models, 0.005, min_dim=896, sort_by='dim', min_zeta=85, max_zeta=92, min_topK=1, max_topK=2)

def search_36B_moe_model():
    models = search_arh_moeLLM(35000000000.0, 6500000000.0, 2, 2.7, 30, 50, (8 / 3), 3)
    return process_models(models, 0.02, min_topK=1, max_topK=2, min_zeta=50, max_zeta=120)

def search_7B_mfa_moe_model():
    for i in range(30, 100):
        print(i)
        models = search_arh_moe_mfa(7000000000.0, ((7000000000.0 * 0.001) * i), 100, 120, 2, 3, 10, 32, 2.75, 128, 128, 50, 100, 1, 0.5, 1, 0.02, 256, 0.5, 0.1, 8192, 65536)
        process_models(models, 0.02, min_topK=1, max_topK=5, min_zeta=50, max_zeta=120)

def search_7B_mfa_moe_model_2():
    for i in range(65, 68):
        print(i)
        models = search_arh_moe_mfa(6513800000.0, ((6513800000.0 * 0.001) * i), 60, 120, 2, 3, 10, 32, 2.75, 128, 128, 10, 100, 1, 0.5, 1, 0.01, 256, 0.5, 0.1, 8192, 65536)
        process_models(models, 0.01, min_topK=1, max_topK=5, min_zeta=50, max_zeta=120)

def search_deepseek_v3():
    for i in range(3, 10):
        print(i)
        models = search_arh_moe_mfa(669170000000.0, ((669170000000.0 * 0.01) * i), 100, 120, 2.0, 3.0, 30, 80, 2.66, 128, 128, 70, 100, 3, 0.4, 1, 0.02, 256, 0.5, 0.01, 8192, 65536)
        process_models(models, 0.02, min_topK=1, max_topK=8, min_zeta=50, max_zeta=120)

def search_36B_mfa_moe_model():
    models = search_arh_moe_mfa(36021000000.0, 6566200000.0, 80, 100, 1, 4, 20, 50, 2.75, 128, 128, 10, 100, 1, 0.7, 0.8, 0.01, 256, 0.5, 0.1, 8192, 65536)
    return process_models(models, 0.01, min_topK=1, max_topK=2, min_zeta=50, max_zeta=120)

def search_500B_moe_model():
    models = search_arh_moeLLM(500000000000.0, 90000000000.0, 2, 2.7, 80, 100, (8 / 3), 3)
    return process_models(models, 0.02, 1, max_topK=4, min_zeta=70, max_zeta=100)

def search_600B_mfa_moe_model():
    models = search_arh_moe_mfa(600000000000.0, 66000000000.0, 70, 120, 2.4, 3, 30, 90, 2.6, 256, 256, 10, 100, 1, 0.7, 0.8, 0.03, 256, 0.5, 0.1, 8192, 65536)
    return process_models(models, 0.03, min_topK=1, max_topK=2, min_zeta=50, max_zeta=120)

def search_300B_mfa_moe_model():
    models = search_arh_moe_mfa(310000000000.0, 36000000000.0, 70, 120, 2.4, 3, 30, 90, 2.6, 256, 256, 10, 100, 1, 0.7, 0.8, 0.03, 256, 0.21, 0.19, 8192, 65536)
    return process_models(models, 0.03, min_topK=3, max_topK=4, min_zeta=50, max_zeta=120)
if (__name__ == '__main__'):
    start_time = time.time()
    search_300B_mfa_moe_model()
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print(f'程序运行时间：{elapsed_time:.6f}秒')
