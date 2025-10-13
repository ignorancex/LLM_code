
import warnings
import itertools
import tqdm
import numpy as np
import pandas as pd
get_N_expr = 'h*numl*(4*h+3*ffnh)'
get_M_expr = '6*(4*h+3*ffnh+2*seq_len)*numl*h'
from scalinglaw_utils.architecture_search.llm_architecture_base import BaseLLMArchitecture

def search_ach_baseLLM(para, emb_ratio, wo_emb_ratio, bound1, bound2, tight_bound, with_emb=True, alpha_low=1.0, alpha_high=100.0, s_fix=2048, head_dim=128, print_out=True, layer_low: int=3, layer_up: int=128):
    print((('Searching ' + '{:.5e}'.format(para)) + ('===' * 50)))
    v_fix = 65536
    (d_model_start, d_model_end) = (head_dim, 20480)
    (l_start, l_end) = (layer_low, layer_up)
    (res_double, res_tight1, res_tight2) = ([], [], [])
    bounds = [(emb_ratio - bound1), (emb_ratio + bound1), (wo_emb_ratio - bound2), (wo_emb_ratio + bound2), (emb_ratio - tight_bound), (emb_ratio + tight_bound), (wo_emb_ratio - tight_bound), (wo_emb_ratio + tight_bound)]
    for layer in range(l_start, l_end):
        for d in range(d_model_start, d_model_end, head_dim):
            n_hat = ((para - ((2 * d) * v_fix)) if with_emb else para)
            dff = ((((n_hat / 3) / d) / layer) - ((4 * d) / 3))
            dff = (int((dff / 8)) * 8)
            if ((dff < d_model_start) or ((dff / d) < alpha_low) or ((dff / d) > alpha_high)):
                continue
            model = BaseLLMArchitecture(s_fix, d, dff, layer, v_fix, head_dim)
            ratio = model.parameter_count().compute_para_ratio()
            ratio2 = model.p_and_c_non_vocab.compute_para_ratio()
            if ((bounds[0] <= ratio < bounds[1]) and (bounds[2] <= ratio2 < bounds[3])):
                res_double.append(model)
            if (bounds[4] <= ratio < bounds[5]):
                if (abs(((model.para_and_compute.para / para) - 1)) <= 0.001):
                    res_tight1.append(model)
            if (bounds[6] <= ratio2 < bounds[7]):
                if (abs(((model.p_and_c_non_vocab.para / para) - 1)) <= 0.001):
                    res_tight2.append(model)
    if print_out:
        print('Ratio tight')
        for r in (res_tight1 if with_emb else res_tight2):
            print(r)
    return (res_tight1 if with_emb else res_tight2)

def search_675():
    search_ach_baseLLM((1073810000.0 * 2), 6.0, 6.75, 0.1, 0.1, 0.1, False, 3.5, 4.5, 2560)

def search_llm_family_2(N_arr: np.array, target_alpha: float=4.0, target_zeta: float=128.0, h_attempt_arr: np.ndarray=np.arange(384, 3072, 64), numl_attempt_arr: np.ndarray=np.arange(3, 32, 1), relax_N_ratio: float=0.005) -> pd.DataFrame:
    '\n        N_arr can be (4**((1/3)* np.arange(0, 10)) * 33554432).astype(int)\n    '
    get_ffnh_max_expr = '(N_max/h/numl-4*h)/3'
    get_ffnh_min_expr = '(N_min/h/numl-4*h)/3'
    get_alpha_expr = 'ffnh/h'
    get_zeta_expr = 'h/numl'
    final_res_df = pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        for N in tqdm.tqdm(N_arr):
            res_df = pd.DataFrame(list(itertools.product(h_attempt_arr, numl_attempt_arr)), columns=['h', 'numl'])
            res_df['N_max'] = (N * (1 + relax_N_ratio))
            res_df['N_min'] = (N * (1 - relax_N_ratio))
            res_df['ffnh_max'] = res_df.eval(get_ffnh_max_expr).apply((lambda x: (8 * round((x / 8))))).astype(int)
            res_df['ffnh_min'] = res_df.eval(get_ffnh_min_expr).apply((lambda x: (8 * round((x / 8))))).astype(int)
            new_res_df = pd.DataFrame()
            for (idx, row) in res_df.iterrows():
                (h, numl, ffnh_max, ffnh_min) = row[['h', 'numl', 'ffnh_max', 'ffnh_min']]
                if (((ffnh_max / h) < 1) or ((ffnh_min / h) > 10) or (numl < 6)):
                    continue
                new_res_df = pd.concat([new_res_df, pd.DataFrame(list(itertools.product([h], [numl], np.arange(ffnh_min, (ffnh_max + 8), 8))), columns=['h', 'numl', 'ffnh'])])
            res_df = new_res_df
            res_df = res_df.query('ffnh/h>1 and ffnh/h<10')
            res_df['N'] = N
            res_df['alpha'] = res_df.eval(get_alpha_expr)
            res_df['zeta'] = res_df.eval(get_zeta_expr)
            res_df['alpha_dev'] = np.abs(((res_df['alpha'] / target_alpha) - 1))
            res_df['zeta_dev'] = np.abs(((res_df['zeta'] / target_zeta) - 1))
            res_df['total_dev_sum'] = (res_df['alpha_dev'] + res_df['zeta_dev'])
            res_df = res_df.sort_values(['total_dev_sum'])
            final_res_df = pd.concat([final_res_df, res_df.iloc[:1]])
    return final_res_df
if (__name__ == '__main__'):
    search_ach_baseLLM((1073810000.0 * 3), 6.0, 6.75, 0.1, 0.1, 1, False, 2.5, 3.1, 2048, 128, True, 15, 40)
