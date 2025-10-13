import math

import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from tabulate import tabulate
from scalinglaw_utils.scaling_law_fiting import non_linear_model
from scalinglaw_utils.scaling_law_fiting.data_filters import calculate_differences, filter_data_further, filter_data
from scalinglaw_utils.scaling_law_fiting.fit_with_bias import fit_with_y_bias, found_best_bias
from scalinglaw_utils.scaling_law_fiting.fit_with_trans import fit_with_x_trans, found_best_factor
from scalinglaw_utils.scaling_law_fiting.fitting_utils import cal_mean_error, rename_columns_with_prefix
from scalinglaw_utils.scaling_law_fiting.group_analysis_multi_value import group_analysis_multi_value, \
    analysis_multi_value
from scalinglaw_utils.scaling_law_fiting.group_fit_warp import group_fit
from scalinglaw_utils.scaling_law_fiting.non_linear_model import PowConConFit
from scalinglaw_utils.scaling_law_fiting.read_data_big_exp import read_data_1222,read_data_bilingual,read_data_code,read_data_val_big_d,read_data_code_73,read_data_code_37,read_data_1222_smooth_loss,read_data_code_73_nl,read_data_code_37_nl,read_data_code_nl
from scaling_law_analysis_power_law import ax_default_setting, plot_dots, plot_dots_nd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker


def scaling_fit_fn_an_bn(df_raw:DataFrame, min_len_g1:int=5, filter_query:str=None, use_pred=True,
                         q_constrain=-0.53, enable_constrain=True, label=0.0, write_out=True, is_plot=False, 
                         fit_variant="all_trans"):
    """
    Scaling law end-to-end fitting with different variants of fitting methods.
    
    Args:
        df_raw: Raw data frame
        min_len_g1: Minimum length of group 1
        filter_query: Query string to filter data
        use_pred: Whether to use prediction
        q_constrain: Q constraint value
        enable_constrain: Whether to enable constraint
        label: Label value
        write_out: Whether to write out the result
        is_plot: Whether to plot the result
        fit_variant: Fitting variant type
            - "all_trans": all rounds use trans fitting (original)
            - "all_power": all rounds use power law fitting
            - "r2_power": r2 uses power law, r1 and r3 use trans fitting
            - "r1r3_power": r1 and r3 use power law, r2 uses trans fitting
    
    Returns:
        df_efn, df_results: The result data frames
    """
    DIFF_RATIO = 2 ** 0.5
    LOG_FACTOR = math.log10(math.e)
    GNAME_N, GNAME_D = "N", "D_round"
    X_D, X_N = "D", "N"
    Y1 = "Minus_L_diff"
    HND, EFN = "Fake_AN_D_BN",  "E_ADD_FN"
    SLOPE, INTER, INTER_RAW, SLOPE_RAW = "slope_abs", "intercept_true", "intercept", "slope"
    FACTOR_RANGE = [-0.5, 0.5, 0.0001, 0]
    FACT_RANGE_2 = [-0.5, 0.5, 0.001, 0]
    BIAS_RANGE = [0, 0.3, 0.001]
    factor_values = np.arange(FACTOR_RANGE[0], FACTOR_RANGE[1], FACTOR_RANGE[2])

    df_trans = df_raw.query(filter_query)
    current_time_str = __import__("datetime").datetime.now().strftime("%y%m%d%H%M%S")

    df = calculate_differences(df_raw, group_by=GNAME_N, sort_by=X_D)
    df = filter_data_further(df, group_name=GNAME_N, diff_filter=X_D + "_diff", min_group_num=min_len_g1)
    if filter_query is not None and len(filter_query) > 0:
        df = df.query(filter_query)
    plot_dots_nd(df_raw, df, "filter_query:" + filter_query)
    # 按照N进行分组拟合，X轴为D，Y轴为 "Minus_L_diff"。得到离散的AN、BN
    df_g_fit, para_g_fit = group_fit(df, group_key=GNAME_N, x_key=X_D, y_key=Y1,
                                     compute_partial_loss=False, x_log=True, y_log=True, min_len=min_len_g1 - 1)
    
    
    # 按照D进行分组拟合，X轴为N，Y轴为 "Minus_L_diff"
    df_g_fit_d, para_g_fit_d = group_fit(df, group_key=GNAME_D, x_key=X_N, y_key=Y1,
                                        compute_partial_loss=False, x_log=True, y_log=True, min_len=min_len_g1 - 1)

    SLOPE_PRED = 'x-' + GNAME_N + "_y-" + SLOPE + "_pred_raw"
    INTER_PRED = 'x-' + GNAME_N + "_y-" + INTER + "_pred_raw"
    para_g_fit[INTER] = (10 ** para_g_fit[INTER_RAW]) / ((DIFF_RATIO ** para_g_fit[SLOPE]) - 1)

    # fit slope first，线性拟合AN，log(AN)=A + aN; A(N) = AN^a + ea; 不加bias ea = 0
    assert len(para_g_fit) > 0, "输入数据 para_g_fit 为空！"
    assert GNAME_N in para_g_fit.columns, f"列 {GNAME_N} 不存在"
    assert SLOPE in para_g_fit.columns, f"列 {SLOPE} 不存在"
    
    # Round 0: Process for slope based on r0_method parameter
    if fit_variant == "r2_power" or fit_variant == "all_power":
        print("\nR0: Using power law fitting for slope with bias search")
        # Search for the best bias value for power law fitting
        best_bias = 0
        min_error = float('inf')
        best_slope_fit = None
        best_para_an_r0 = None
        
        for bias in factor_values:
        # for bias in [0]:
            if abs(bias) < FACTOR_RANGE[3]:
                continue
            if para_g_fit[SLOPE].min() <= bias:
                continue
            
            df_tmp, para_tmp = fit_with_y_bias(para_g_fit, GNAME_N, SLOPE, bias, True, True)
            
            # Basic error calculation to determine best fit
            rel_error = para_tmp["relative_residual_mean"].item()
            
            if rel_error < min_error:
                min_error = rel_error
                best_bias = bias
                best_slope_fit = df_tmp
                best_para_an_r0 = para_tmp
        
        slope_fit_power, para_an_r0 = best_slope_fit, best_para_an_r0
        print(f"Best bias for r0 power law fitting: {best_bias}, error: {min_error}")
    else:
        print("\nR0: Using trans (exponential) fitting for slope")
        # Search for best factor for trans fitting
        best_factor = 0
        min_error = float('inf')
        best_slope_fit = None
        best_para_an_r0 = None
        
        for factor in factor_values:
            if abs(factor) < FACTOR_RANGE[3]:
                continue
            
            df_tmp, para_tmp = fit_with_x_trans(para_g_fit, GNAME_N, SLOPE, factor)
            
            # Basic error calculation
            rel_error = para_tmp["relative_residual_mean"].item()
            
            if rel_error < min_error:
                min_error = rel_error
                best_factor = factor
                best_slope_fit = df_tmp
                best_para_an_r0 = para_tmp
        
        slope_fit_power, para_an_r0 = best_slope_fit, best_para_an_r0
        print(f"Best factor for r0 trans fitting: {best_factor}, error: {min_error}")
         
    print(tabulate(slope_fit_power, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.7f'))

    # 拟合B(N), exp(BN^b+Q), z=N^b, B(N)=exp(Bz+Q),
    # 按照上述的slope选择一个最好的 Intercept bias
    hnd_diff_r_error_r1_min, min_inter_factor_r1, df_inter_r1, para_bn_r1 = float('inf'), float('inf'), None, None
    hnd_diff_r_error_best_r1, best_inter_factor_r1, df_inter_r1_b, para_bn_r1_b = float('inf'), float('inf'), None, None
    an_d_bn_diff_r_error_list, intercept_fit_relative_error_list = [], []
    
    # Round 1: Process for intercept based on the variant
    if fit_variant in ["all_power", "r1r3_power"]:
        print("\nR1: Using power law fitting for intercept")
        # Search for the best bias value for power law fitting
        df_inter_fits, fit_res_list = [], []
        min_error, best_bias = float('inf'), 0
        
        for bias in factor_values:
        # for bias in [0]:
            if abs(bias) < FACTOR_RANGE[3]:
                continue
            if slope_fit_power[INTER].min() <= bias:
                continue
            df_inter_tmp, para_bn_tmp = fit_with_y_bias(slope_fit_power, GNAME_N, INTER, bias, True, True)
            
            # Calculate error metrics
            df_tmp = df_trans.merge(df_inter_tmp[[GNAME_N, SLOPE_PRED if use_pred else SLOPE, INTER_PRED]], 
                                    on=GNAME_N, how='inner')
            df_tmp["Fake_AN_D_BN"] = (df_tmp[INTER_PRED] * 
                                    (df_tmp[X_D] ** -df_tmp[SLOPE_PRED if use_pred else SLOPE]))
            df_tmp_diff = calculate_differences(df_tmp, group_by=GNAME_N, sort_by=X_D, v_keys=["L", "Fake_AN_D_BN"])
            df_tmp_res = filter_data_further(df_tmp_diff, group_name=GNAME_N, diff_filter=X_D + "_diff",
                                            min_group_num=min_len_g1, drop_nan_key="Fake_AN_D_BN_diff")
            _, error = cal_mean_error(df_tmp_res, Y1, "Minus_Fake_AN_D_BN_diff")
            
            df_inter_fits.append(df_inter_tmp)
            fit_res_list.append(para_bn_tmp)
            
            if error < min_error:
                min_error = error
                best_bias = bias
                df_inter_r1 = df_inter_tmp
                para_bn_r1 = para_bn_tmp
        
        # print(f"Best bias for power law intercept fitting: {best_bias}, error: {min_error}")
        best_inter_factor_r1 = best_bias
        min_inter_factor_r1 = best_bias
        df_inter_r1_b = df_inter_r1.copy()
        para_bn_r1_b = para_bn_r1.copy()
        hnd_diff_r_error_r1_min = min_error
        hnd_diff_r_error_best_r1 = min_error
    else:
        print("\nR1: Using trans fitting for intercept")
        # Original code for trans fitting search
        # 对于每一个factor(b)，做一次线性拟合，得到B、Q。得到了B(N)。加上A(N)，反推出h(N,D)=B(N)D^A(N) -->"Fake_AN_D_BN"
        # 用"Fake_AN_D_BN"，做差分法，把差分之后的Minus_Fake_AN_D_BN_diff，和原有的真实的差值来计算loss，
        # 求loss最小的b；
        for factor in factor_values:
            if abs(factor) < FACTOR_RANGE[3]:
                continue
            df_tmp_anbn_r1, para_tmp_inter_r1 = fit_with_x_trans(slope_fit_power, GNAME_N, INTER, factor)
            
            df_tmp_r1 = df_trans.merge(df_tmp_anbn_r1[[GNAME_N, SLOPE_PRED if use_pred else SLOPE, INTER_PRED]],
                                     on=GNAME_N, how='inner')
            df_tmp_r1["Fake_AN_D_BN"] = (df_tmp_r1[INTER_PRED] *
                                         (df_tmp_r1[X_D] ** -df_tmp_r1[SLOPE_PRED if use_pred else SLOPE]))
            df_tmp_r1_diff = calculate_differences(df_tmp_r1, group_by=GNAME_N, sort_by=X_D, v_keys=["L", "Fake_AN_D_BN"])
            df_tmp_r1_res = filter_data_further(df_tmp_r1_diff, group_name=GNAME_N, diff_filter=X_D + "_diff",
                                            min_group_num=min_len_g1, drop_nan_key="Fake_AN_D_BN_diff")
            intercept_fit_relative_error_list.append(para_tmp_inter_r1["relative_residual_mean"].item())
            hnd_diff_error, an_d_bn_diff_r_error = cal_mean_error(df_tmp_r1_res, Y1, "Minus_Fake_AN_D_BN_diff")
            an_d_bn_diff_r_error_list.append(an_d_bn_diff_r_error)
            # 确保B>0（SLOPE_RAW>0）和b<0（factor<0）
            if an_d_bn_diff_r_error < hnd_diff_r_error_r1_min and para_tmp_inter_r1[SLOPE_RAW].item() > 0 and factor < 0: # 更改bB限制
                hnd_diff_r_error_r1_min = an_d_bn_diff_r_error
                min_inter_factor_r1, df_inter_r1, para_bn_r1 = factor, df_tmp_anbn_r1, para_tmp_inter_r1
            if (para_tmp_inter_r1[SLOPE_RAW].item() > 0 and factor < 0  and para_tmp_inter_r1[INTER_RAW].item() < q_constrain):
                if an_d_bn_diff_r_error < hnd_diff_r_error_best_r1:
                    hnd_diff_r_error_best_r1 = an_d_bn_diff_r_error
                    best_inter_factor_r1, df_inter_r1_b, para_bn_r1_b = factor, df_tmp_anbn_r1, para_tmp_inter_r1

    print(f'min intercept factor:{min_inter_factor_r1}, the relative error:{hnd_diff_r_error_r1_min}, '
          f'slope:{para_bn_r1[SLOPE_RAW].item()}, intercept:{para_bn_r1[INTER_RAW].item()/LOG_FACTOR}')
    if fit_variant not in ["all_power", "r1r3_power"] and hnd_diff_r_error_best_r1 < float('inf'):
        print(f'best intercept factor:{best_inter_factor_r1}, the relative error:{hnd_diff_r_error_best_r1}, '
              f'slope:{para_bn_r1_b[SLOPE_RAW].item()}, intercept:{para_bn_r1_b[INTER_RAW].item()/LOG_FACTOR}')
        if enable_constrain:
            min_inter_factor_r1, df_inter_r1, para_bn_r1  = best_inter_factor_r1, df_inter_r1_b, para_bn_r1_b

    df_inter_r1 = rename_columns_with_prefix(df_inter_r1, 'x-' + GNAME_N + "_y-" + SLOPE , "_r0")
    print(tabulate(df_inter_r1, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.7f'))

    # Round 2: Process for slope based on the variant
    hnd_diff_r_error_r2_min, min_slope_factor_r2, df_slope_r2, para_an_r2 = float('inf'), float('inf'), None, None
    an_d_bn_diff_r_error_list = []
    
    if fit_variant in ["all_power", "r2_power"]:
        print("\nR2: Using power law fitting for slope")
        # Search for best bias for power law fitting
        best_bias = 0
        min_error = float('inf')
        df_slope_r2_tmp = None
        para_an_tmp = None
        
        for bias_value in factor_values:
        # for bias_value in [0]:
            if abs(bias_value) < FACTOR_RANGE[3]:
                continue
            if df_inter_r1[SLOPE].min() <= bias_value:
                continue
            df_tmp, para_tmp = fit_with_y_bias(df_inter_r1, GNAME_N, SLOPE, bias=float(bias_value), x_log=True, y_log=True)
            
            # Calculate error metrics
            df_tmp_r2 = df_trans.merge(df_tmp[[GNAME_N, SLOPE_PRED, INTER_PRED]], on=GNAME_N, how='inner')
            df_tmp_r2["Fake_AN_D_BN"] = df_tmp_r2[INTER_PRED] * (df_tmp_r2[X_D] ** -df_tmp_r2[SLOPE_PRED])
            df_tmp_r2_diff = calculate_differences(df_tmp_r2, group_by=GNAME_N, sort_by=X_D, v_keys=["L", "Fake_AN_D_BN"])
            df_tmp_r2_res = filter_data_further(df_tmp_r2_diff, group_name=GNAME_N, diff_filter=X_D + "_diff",
                                            min_group_num=min_len_g1, drop_nan_key="Fake_AN_D_BN_diff")
            _, current_error = cal_mean_error(df_tmp_r2_res, Y1, "Minus_Fake_AN_D_BN_diff")
            
            if current_error < min_error:
                min_error = current_error
                best_bias = bias_value
                df_slope_r2_tmp = df_tmp
                para_an_tmp = para_tmp
        
        df_slope_r2 = df_slope_r2_tmp
        para_an_r2 = para_an_tmp
        min_slope_factor_r2 = best_bias
        hnd_diff_r_error_r2_min = min_error
    else:
        print("\nR2: Using trans fitting for slope")
        # Original code for trans fitting
        for factor in factor_values:
            if abs(factor) < FACTOR_RANGE[3]:
                continue
            df_tmp_anbn_r2, para_tmp_slope_r2 = fit_with_x_trans(df_inter_r1, GNAME_N, SLOPE, factor)
            df_tmp_r2 = df_trans.merge(df_tmp_anbn_r2[[GNAME_N, SLOPE_PRED, INTER_PRED]], on=GNAME_N, how='inner')
            df_tmp_r2["Fake_AN_D_BN"] = df_tmp_r2[INTER_PRED] * (df_tmp_r2[X_D] ** -df_tmp_r2[SLOPE_PRED])
            df_tmp_r2_diff = calculate_differences(df_tmp_r2, group_by=GNAME_N, sort_by=X_D,
                                                 v_keys=["L", "Fake_AN_D_BN"])
            df_tmp_r2_res = filter_data_further(df_tmp_r2_diff, group_name=GNAME_N, diff_filter=X_D + "_diff",
                                            min_group_num=min_len_g1, drop_nan_key="Fake_AN_D_BN_diff")
            _, hnd_diff_r_error = cal_mean_error(df_tmp_r2_res, Y1, "Minus_Fake_AN_D_BN_diff")
            an_d_bn_diff_r_error_list.append(hnd_diff_r_error)
            if hnd_diff_r_error < hnd_diff_r_error_r2_min:
                hnd_diff_r_error_r2_min = hnd_diff_r_error
                min_slope_factor_r2, para_an_r2, df_slope_r2 = factor, para_tmp_slope_r2, df_tmp_anbn_r2

    print(f'min slope factor is {min_slope_factor_r2}, the relative error is {hnd_diff_r_error_r2_min}, '
          f'slope: {para_an_r2[SLOPE_RAW].item()}, intercept: {para_an_r2[INTER_RAW].item()/LOG_FACTOR}')

    df_slope_r2 = rename_columns_with_prefix(df_slope_r2, 'x-' + GNAME_N + "_y-" + INTER , "_r1")
    print(tabulate(df_slope_r2, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.7f'))

    # Round 3: Process for intercept based on the variant
    hnd_diff_r_error_r3_min, min_inter_factor_r3, df_inter_r3, para_bn_r3 = float('inf'), float('inf'), None, None
    
    if fit_variant in ["all_power", "r1r3_power"]:
        print("\nR3: Using power law fitting for intercept")
        # Search for best bias for power law fitting
        best_bias = 0
        min_error = float('inf')
        df_inter_r3_tmp = None
        para_bn_tmp = None
        
        for bias_value in factor_values:
        # for bias_value in [0]:
            if abs(bias_value) < FACTOR_RANGE[3]:
                continue
            if df_slope_r2[SLOPE].min() <= bias_value:
                continue
            df_tmp, para_tmp = fit_with_y_bias(df_slope_r2, GNAME_N, INTER, bias=float(bias_value), x_log=True, y_log=True)
            
            # Calculate error metrics
            df_tmp_r3 = df_trans.merge(df_tmp[[GNAME_N, SLOPE_PRED, INTER_PRED]], on=GNAME_N, how='inner')
            df_tmp_r3["Fake_AN_D_BN"] = df_tmp_r3[INTER_PRED] * (df_tmp_r3[X_D] ** -df_tmp_r3[SLOPE_PRED])
            df_tmp_r3_diff = calculate_differences(df_tmp_r3, group_by=GNAME_N, sort_by=X_D, v_keys=["L", "Fake_AN_D_BN"])
            df_tmp_r3_diff = filter_data_further(df_tmp_r3_diff, group_name=GNAME_N, diff_filter=X_D + "_diff",
                                            min_group_num=min_len_g1, drop_nan_key="Fake_AN_D_BN_diff")
            _, current_error = cal_mean_error(df_tmp_r3_diff, Y1, "Minus_Fake_AN_D_BN_diff")
            
            if current_error < min_error:
                min_error = current_error
                best_bias = bias_value
                df_inter_r3_tmp = df_tmp
                para_bn_tmp = para_tmp
        
        df_inter_r3 = df_inter_r3_tmp
        para_bn_r3 = para_bn_tmp
        min_inter_factor_r3 = best_bias
        hnd_diff_r_error_r3_min = min_error
    else:
        print("\nR3: Using trans fitting for intercept")
        # Original code for trans fitting
        for factor in factor_values:
            if abs(factor) < FACTOR_RANGE[3]:
                continue
            df_tmp_anbn_r3, para_tmp_inter_r3 = fit_with_x_trans(df_slope_r2, GNAME_N, INTER, factor)
            df_tmp_r3 = df_trans.merge(df_tmp_anbn_r3[[GNAME_N, SLOPE_PRED, INTER_PRED]], on=GNAME_N, how='inner')
            df_tmp_r3["Fake_AN_D_BN"] = df_tmp_r3[INTER_PRED] * (df_tmp_r3[X_D] ** -df_tmp_r3[SLOPE_PRED])
            df_tmp_r3_diff = calculate_differences(df_tmp_r3, group_by=GNAME_N, sort_by=X_D,
                                                 v_keys=["L", "Fake_AN_D_BN"])
            df_tmp_r3_diff = filter_data_further(df_tmp_r3_diff, group_name=GNAME_N, diff_filter=X_D + "_diff",
                                            min_group_num=min_len_g1, drop_nan_key="Fake_AN_D_BN_diff")
            _, hnd_diff_r_error = cal_mean_error(df_tmp_r3_diff, Y1, "Minus_Fake_AN_D_BN_diff")
            # 对min_inter_factor_r3也应用相似的约束，确保B>0和b<0
            if hnd_diff_r_error < hnd_diff_r_error_r3_min and para_tmp_inter_r3[SLOPE_RAW].item() > 0 and factor < 0: # 更改bB限制
                hnd_diff_r_error_r3_min = hnd_diff_r_error
                min_inter_factor_r3, para_bn_r3, df_inter_r3 = factor, para_tmp_inter_r3, df_tmp_anbn_r3

    print(f'min intercept factor is {min_inter_factor_r3}, the relative error is {hnd_diff_r_error_r3_min}, '
          f'slope: {para_bn_r3[SLOPE_RAW].item()}, intercept: {para_bn_r3[INTER_RAW].item()/LOG_FACTOR}')
    print(tabulate(df_inter_r3, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.7f'))
    SLOPE_PRED0, INTER_PRED1 = SLOPE_PRED+"_r0", INTER_PRED+"_r1"
    if is_plot:
        analysis_multi_value(df_inter_r3, [SLOPE_PRED0, SLOPE_PRED,INTER_PRED1, INTER_PRED], X_N,
                         [SLOPE,SLOPE,INTER,INTER])

    HND_TITLE = ["R2_slope,R3_inter"]
    JOIN_KEY = [[SLOPE_PRED, INTER_PRED]]
    df_efn = df.merge(df_inter_r3[[GNAME_N, SLOPE, INTER, SLOPE_PRED, INTER_PRED, SLOPE_PRED0, INTER_PRED1]], on=GNAME_N, how='inner')
    print(f'line:{df_g_fit.shape[0]}->{df_g_fit.dropna().shape[0]}->{df_efn.dropna().shape[0]}')
    enf_factors,enf_biases,efn_para = dict(), dict(), dict()
    # print(df_efn.columns)
    for i in range(len(HND_TITLE)):
        efn_key, hnd_key = EFN + ("" if  i == 0 else f'{i}'), HND + ("" if  i == 0 else f'{i}')
        df_efn[hnd_key] = df_efn[JOIN_KEY[i][1]] * (df_efn[X_D] ** -df_efn[JOIN_KEY[i][0]])
        # print(df_efn[hnd_key].mean())
        # print(hnd_key, JOIN_KEY[i][0], JOIN_KEY[i][1])
        df_efn[efn_key] = df_efn["L"] - df_efn[hnd_key]
        df_efn[efn_key+"_mean"] = df_efn.groupby(GNAME_N)[efn_key].transform('mean')
        enf_factor, re1 = found_best_factor(df_efn, GNAME_N, efn_key, FACT_RANGE_2, is_plot=False, is_print=False)
        # enf_bias, re2 = found_best_bias(df_efn, GNAME_N, efn_key, BIAS_RANGE, is_plot=False, is_print=False)
        enf_bias = 0
        re2 = 0
        enf_factors[HND_TITLE[i]] = enf_factor
        enf_biases[HND_TITLE[i]] = enf_bias
        print(f'Best factor for {HND_TITLE[i]}: {enf_factor}, r error: {re1}, best bias: {enf_bias}, r error: {re2}')
        df_efn, para_efn = fit_with_x_trans(df_efn, GNAME_N, efn_key, enf_factor)
        df_efn, para_efn_p = fit_with_y_bias(df_efn, GNAME_N, efn_key, enf_bias, suffix="_p")
        efn_para[HND_TITLE[i]] = [para_efn, para_efn_p]
    results = []
    param_keys = ['E', 's', 'q', 'S', 'B', 'b', 'Q', 'A', 'a']
    d_or_c, e_or_p = ["_c"], ["", "_p"]
    min_end2end_r_error, min_index, p_min_i, p_index = float('inf'), (-1,-1,-1), 0, 0
    for i in range(len(HND_TITLE)):
        efn_key, hnd_key = EFN + ("" if  i == 0 else f'{i}'), HND + ("" if  i == 0 else f'{i}')
        for j in range(len(d_or_c)):
            s1 = d_or_c[j]
            use_enf = efn_key if s1=="_c" else EFN
            for k in range(len(e_or_p)):
                s2 = e_or_p[k]
                enf_pred = f'x-N_y-{use_enf}{s2}_pred_raw'
                para_all = {'E': 0, 's': None, 'q': None, 'S': None,
                            'B': None, 'b': None, 'Q': None, 'A': None, 'a': None}
                df_efn[f"L_pred{i}{s1}{s2}"] = df_efn[hnd_key] + df_efn[f'x-N_y-{use_enf}{s2}_pred_raw']
                abs_error_all, abs_r_error_all = cal_mean_error(df_efn, "L", f"L_pred{i}{s1}{s2}")
                abs_error_efn, abs_r_error_efn = cal_mean_error(df_efn, use_enf, enf_pred)
                abs_error_var, abs_r_error_var = cal_mean_error(df_efn, use_enf,f'{use_enf}_mean')
                abs_error_fit, abs_r_error_fit = cal_mean_error(df_efn, f'{use_enf}_mean', enf_pred)
                # 解析出所有的参数, s2 决定 fn的形式， HND_TITLE[i][0] 决定 an 的形式
                # (self.E, self.s, self.q, self.S, self.B, self.b, self.Q, self.A, self.a)
                enf_key = HND_TITLE[i] if j == 1 else HND_TITLE[0]
                if s2 == "_p":
                    para_all['q'] = efn_para[enf_key][1][SLOPE_RAW].item()
                    para_all['S'] = enf_biases[enf_key]
                    para_all['s'] = 10 ** efn_para[enf_key][1][INTER_RAW].item()
                else:
                    para_all['s'] = efn_para[enf_key][0][SLOPE_RAW].item() / LOG_FACTOR
                    para_all['q'] = enf_factors[enf_key]
                    para_all['S'] = efn_para[enf_key][0][INTER_RAW].item() / LOG_FACTOR

                if fit_variant in ["all_power", "r2_power"]:
                    para_all['a'] = para_an_r2[SLOPE_RAW].item()
                    para_all['E'] = min_slope_factor_r2
                    para_all['A'] = 10 ** para_an_r2[INTER_RAW].item()
                else:
                    para_all['A'] = para_an_r2[SLOPE_RAW].item() / LOG_FACTOR
                    para_all['a'] = min_slope_factor_r2
                    para_all['E'] = para_an_r2[INTER_RAW].item() / LOG_FACTOR

                if fit_variant in ["all_power", "r1r3_power"]:
                    para_all['b'] = para_bn_r3[SLOPE_RAW].item()
                    para_all['Q'] = min_inter_factor_r3
                    para_all['B'] = 10 ** para_bn_r3[INTER_RAW].item()
                else:
                    para_all['B'] = para_bn_r3[SLOPE_RAW].item() / LOG_FACTOR
                    para_all['b'] = min_inter_factor_r3
                    para_all['Q'] = para_bn_r3[INTER_RAW].item() / LOG_FACTOR
                
                # para_all['B'] = para_bn_final[SLOPE_RAW].item() / LOG_FACTOR
                # para_all['b'] = bn_factor
                # para_all['Q'] = para_bn_final[INTER_RAW].item() / LOG_FACTOR

                # para_all['A'] = para_an_r2[SLOPE_RAW].item() / LOG_FACTOR
                # para_all['a'] = min_slope_factor_r2
                # para_all['E'] = para_an_r2[INTER_RAW].item() / LOG_FACTOR
                # fit_class_name = (("Exp" if s2 == "" else "Pow") + "Exp" +
                #                   ("Exp" if JOIN_KEY[i][0] == SLOPE_PRED else "Pow") + "Fit")
                if s2 == "_p":
                    part1 = "Pow"
                else:
                    part1 = "Exp"
                if fit_variant == "all_power":
                    part2 = "Pow"
                    part3 = "Pow"
                elif fit_variant == "r2_power":
                    part2 = "Exp"
                    part3 = "Pow"
                elif fit_variant == "r1r3_power":
                    part2 = "Pow"
                    part3 = "Exp"
                else:
                    part2 = "Exp"
                    part3 = "Exp"
                fit_class_name = part1 + part2 + part3 + "Fit"
                fit_class = getattr(non_linear_model, fit_class_name, None)
                this_fit = fit_class(params_dict=para_all)
                df_efn = this_fit.batch_pred(df_efn, 'N', 'D', f"L_pred{i}{s1}{s2}_2", False)
                # cal_mean_error(df_efn, "L", f"L_pred{i}{s1}{s2}", True)
                abs_error_rerun, abs_r_error_rerun = cal_mean_error(df_efn, "L", f"L_pred{i}{s1}{s2}_2")
                # cal_mean_error(df_efn, f"L_pred{i}{s1}{s2}", f"L_pred{i}{s1}{s2}_2", True)
                results.append([current_time_str, filter_query, f"L_pred{i}{s1}{s2}", f"{HND_TITLE[i]}_{fit_variant}", abs_error_rerun,
                    abs_r_error_rerun, fit_class_name,  abs_error_all, abs_r_error_all, abs_error_efn, abs_r_error_efn,
                    abs_error_var, abs_r_error_var, abs_error_fit, abs_r_error_fit, label] + [para_all[key] for key in param_keys])

                if abs_r_error_all < min_end2end_r_error:
                    min_end2end_r_error = abs_r_error_all
                    min_index = (i, j, k)
                    p_min_i = p_index
                p_index = p_index + 1
    #                 # (self.E, self.s, self.q, self.S, self.B, self.b, self.Q, self.A, self.a)
    df_results = DataFrame(results, columns=[
        'fit_time', 'fit_query', 'Pred_name', 'fit_method', 'Rerun A-Error', 'Rerun AR-Error', 'fit_class', 'All Error',
        'All AR-Error', 'E+fn A-Error', 'E+fn AR-Error', 'FN-Inner A-Error', 'FN-Inner AR-Error',
        'FN-Fit A-Error', 'FN-Fit AR-Error', 'label'] + param_keys)

    print(tabulate(
        [[f'{cell:.7f}' if isinstance(cell, float) else cell for cell in row] for row in df_results.values.tolist()],
        headers=df_results.columns, tablefmt='pretty', showindex=False))
    if is_plot:
        unique_values = df_results['Pred_name'].unique()
        for unique_value in unique_values:
            if unique_value.find('L_pred1') >= 0 and unique_value.find('L_pred2') >= 0:
                continue
            group_analysis_multi_value(df_efn, [unique_value], X_D, 'L',
                                       GNAME_N, [GNAME_N, GNAME_D, "D_N_bin_int"], True)
        
    if write_out:
        df_efn.to_csv(f'./model/res_codenl_{fit_variant}_{current_time_str}.csv', index=False)
        df_results.to_csv(f'./model/fit_codenl_{fit_variant}_{current_time_str}.csv', index=False)
    return df_efn, df_results


def scaling_fit_fn_gd(df_raw, min_len_g1:int=5, filter_query:str=None, label=0.0, write_out=True, is_plot=True, prefix=""):
    DIFF_RATIO = 2 ** 0.5
    GNAME_N, GNAME_D = "N", "D_round"
    X_D, X_N = "D", "N"
    Y1 = "Minus_L_diff"
    SLOPE, INTER, INTER_RAW, SLOPE_RAW = "slope_abs", "intercept_true", "intercept", "slope"
    results = []
    current_time_str = __import__("datetime").datetime.now().strftime("%y%m%d%H%M%S")

    df_trans = df_raw.query(filter_query)

    df_gd = calculate_differences(df_raw, group_by=GNAME_N, sort_by=X_D)
    df_gd = filter_data_further(df_gd, group_name=GNAME_N, diff_filter=X_D + "_diff", min_group_num=min_len_g1)

    df_fn = calculate_differences(df_raw, group_by=GNAME_D, sort_by=X_N)
    df_fn = filter_data_further(df_fn, group_name=GNAME_D, diff_filter=X_N + "_diff", min_group_num=min_len_g1)

    if filter_query is not None and len(filter_query) > 0:
        df_gd = df_gd.query(filter_query)
        df_fn = df_fn.query(filter_query)

    # B(D/x)^b - B(D)^b = BD^b * (1/x^b - 1)
    # hat(B) = log10(B(1/x^b - 1))  = log10(B(x^-b-1))
    # B = 10 ** (\hat(B)) / (x^-b-1)
    fit_res_gd, para_gd = fit_with_y_bias(df_gd, X_D, Y1, 0, True, True)
    fit_res_fn, para_fn = fit_with_y_bias(df_fn, X_N, Y1, 0, True, True)
    para_gd[INTER] =  (10 ** para_gd[INTER_RAW]) / ((DIFF_RATIO ** para_gd[SLOPE]) - 1)
    para_fn[INTER] =  (10 ** para_fn[INTER_RAW]) / ((DIFF_RATIO ** para_fn[SLOPE]) - 1)
    param_keys = ['E', 's', 'q', 'S', 'B', 'b', 'Q', 'A', 'a']
    para = {'E':0, 's':0, 'q':0, 'S':0, 'B':para_gd[INTER].item(), 'b':para_gd[SLOPE_RAW].item(),
            'Q':0, 'A':para_fn[INTER].item(), 'a':para_fn[SLOPE_RAW].item()}
    fit_wo_bias = PowConConFit(para)
    df_tmp = fit_wo_bias.batch_pred(df_trans, 'N','D','FN_GD')
    df_tmp['ir_loss'] = df_tmp['L'] - df_tmp['FN_GD']
    df_tmp["ir_loss_mean"] = df_tmp['ir_loss'].mean()
    ir_loss = df_tmp['ir_loss'].mean()
    para_up = para.copy()
    para_up['E'] = ir_loss

    df_tmp['L_pred_1'] = para_up['E'] + df_tmp['FN_GD']
    fit_w_bias = PowConConFit(para_up)
    df_res = fit_w_bias.batch_pred(df_tmp, 'N','D','L_pred_2')
    error_1, r_error_1 = cal_mean_error(df_res, "L", "L_pred_1", True)
    error_2, r_error_2 = cal_mean_error(df_res, "L", "L_pred_2", True)
    error_3, r_error_3 = cal_mean_error(df_res, "ir_loss", "ir_loss_mean", True)

    results.append([current_time_str, filter_query, "Chinchilla", 'Default', error_2,
                    r_error_2, 'PowConConFit', 0,0,0,0, error_3, r_error_3,0,0,label]
                   + [para_up[key] for key in param_keys])
    df_results = DataFrame(results, columns=['fit_time', 'fit_query', 'Pred_name', 'fit_method', 'Rerun A-Error',
                                             'Rerun AR-Error', 'fit_class', 'All Error', 'All AR-Error', 'E+fn A-Error',
                                             'E+fn AR-Error', 'FN-Inner A-Error', 'FN-Inner AR-Error', 'FN-Fit A-Error',
                                             'FN-Fit AR-Error', 'label'] + param_keys)

    print(tabulate(
        [[f'{cell:.7f}' if isinstance(cell, float) else cell for cell in row] for row in df_results.values.tolist()],
        headers=df_results.columns, tablefmt='pretty', showindex=False))
    if is_plot:
        group_analysis_multi_value(df_res, ["L_pred_1"], X_D, 'L',
                                   GNAME_N, [GNAME_N, GNAME_D, "D_N_bin_int"], True)
    if write_out:
        df_res.to_csv(f'./model/{prefix}_res_{current_time_str}.csv', index=False)
        df_results.to_csv(f'./model/{prefix}_fit_{current_time_str}.csv', index=False)
    return df_res, df_results

def scaling_fit_torch(df_raw, min_len_g1:int=5, filter_query:str=None, label=0.0, write_out=True, is_plot=True, prefix=""):
    """使用PyTorch拟合Chinchilla缩放律的参数。
    
    Args:
        df_raw: 原始数据框
        min_len_g1: 最小组长度
        filter_query: 过滤查询
        label: 标签
        write_out: 是否写出结果
        is_plot: 是否绘图
        prefix: 输出文件前缀
    
    Returns:
        df_res, df_results: 处理后的数据框和拟合结果
    """
    current_time_str = __import__("datetime").datetime.now().strftime("%y%m%d%H%M%S")
    param_keys = ['E', 's', 'q', 'S', 'B', 'b', 'Q', 'A', 'a']
    results = []
    
    # 过滤数据
    df_trans = df_raw.query(filter_query) if filter_query else df_raw
    
    # 构建PyTorch输入张量
    inp = torch.Tensor([
        [row["N"], row["D"], row["L"]] for _, row in df_trans.iterrows()
    ])
    inp.requires_grad = True

    def loss(inp, params, tie):
        if tie:
            a, b, e, alpha, beta = params[0], params[1], params[2], params[3], params[3]
        else:
            a, b, e, alpha, beta = params[0], params[1], params[2], params[3], params[4]

        pre_lse = torch.stack([a - alpha*torch.log(inp[:, 0]), b - beta*torch.log(inp[:, 1]), e.expand((inp.shape[0]))])
        post_lse = torch.logsumexp(pre_lse, dim=0)
        huber_loss = torch.nn.functional.huber_loss(post_lse, torch.log(inp[:, 2]), delta=1e-3, reduction='none')
        return huber_loss.sum()

    def minimize_loss(inp, init_params=[6, 6, -1, 0.28, 0.28], steps=50, fixed=False, fix_e=False, tie=False):
        params = torch.nn.Parameter(data=torch.Tensor(init_params))
        if tie:
            params = torch.nn.Parameter(data=torch.Tensor(init_params[:-1]))

        lbfgs = torch.optim.LBFGS([params],
                        lr=1e-1,
                        history_size=10,
                        max_iter=20,
                        line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            l = loss(inp, params, tie=tie)
            l.backward()
            if fixed:
                params.grad[:3] = 0
            elif fix_e:
                params.grad[2] = 0
            return l

        for i in range(steps):
            l = lbfgs.step(closure)
        return l, params

    min_loss = 1e10
    best_params = None

    # 网格搜索最优参数
    for a in np.linspace(0, 25, 5):
        for b in np.linspace(0, 25, 5):
            for e in np.linspace(-1, 1, 4):
                for alpha in np.linspace(0, 2, 4):
                    for beta in np.linspace(0, 2, 4):
                        l, params = minimize_loss(inp, [a, b, e, alpha, beta], tie=False)
                        if l < min_loss:
                            min_loss = l
                            best_params = params.detach().numpy()

    # 解析最优参数
    a, b, e, alpha, beta = list(best_params)
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    
    # 转换为与scaling_fit_fn_gd兼容的参数格式
    para = {'E': E, 's': 0, 'q': 0, 'S': 0, 
            'B': B, 'b': -beta, 'Q': 0, 
            'A': A, 'a': -alpha}
    
    # 使用相同的PowConConFit模型进行预测
    fit_wo_bias = PowConConFit(para)
    df_tmp = fit_wo_bias.batch_pred(df_trans, 'N', 'D', 'FN_GD')
    df_tmp['L_pred_1'] = df_tmp['FN_GD']  # 使用拟合后的模型预测
    
    # 计算误差
    error_2, r_error_2 = cal_mean_error(df_tmp, "L", "L_pred_1", True)
    
    # 保存结果
    results.append([current_time_str, filter_query, "Chinchilla_torch", 'PyTorch', error_2,
                    r_error_2, 'PowConConFit', 0, 0, 0, 0, 0, 0, 0, 0, label]
                   + [para[key] for key in param_keys])
    
    df_results = DataFrame(results, columns=['fit_time', 'fit_query', 'Pred_name', 'fit_method', 'Rerun A-Error',
                                             'Rerun AR-Error', 'fit_class', 'All Error', 'All AR-Error', 'E+fn A-Error',
                                             'E+fn AR-Error', 'FN-Inner A-Error', 'FN-Inner AR-Error', 'FN-Fit A-Error',
                                             'FN-Fit AR-Error', 'label'] + param_keys)

    print(tabulate(
        [[f'{cell:.7f}' if isinstance(cell, float) else cell for cell in row] for row in df_results.values.tolist()],
        headers=df_results.columns, tablefmt='pretty', showindex=False))
    
    if is_plot:
        group_analysis_multi_value(df_tmp, ["L_pred_1"], "D", 'L',
                                   "N", ["N", "D_round", "D_N_bin_int"], True)
    
    if write_out:
        df_tmp.to_csv(f'./model/{prefix}_res_torch_{current_time_str}.csv', index=False)
        df_results.to_csv(f'./model/{prefix}_fit_torch_{current_time_str}.csv', index=False)
    
    return df_tmp, df_results

if __name__ == '__main__':
    import time
    start_time = time.time()
    # df_1222= read_data_1222()
    # df_trans = filter_data(df_1222, min_D=1)

    df_code = read_data_code_nl()
    df_trans = filter_data(df_code, min_D=1)
    
    # 运行所有变种
    filter_query = "2e8<N<4e9"
    min_len = 5
    label = 4e9
    
    # scaling_fit_torch(df_trans, 5, filter_query, 7e9, True, False, prefix="Chinchilla_torch")

    # scaling_fit_fn_an_bn(df_trans, 5, f"2e8<N<{label}", True, 2-0.53, False, label, True, False)
    scaling_fit_fn_an_bn(df_trans, min_len, filter_query, use_pred=False, label=label, enable_constrain=False, fit_variant="all_trans")
    
    # scaling_fit_fn_an_bn(df_trans, min_len, filter_query, use_pred=False, label=label, enable_constrain=False, fit_variant="all_power")

    # scaling_fit_fn_an_bn(df_trans, min_len, filter_query, use_pred=False, label=label, enable_constrain=False, fit_variant="r2_power")
    
    # scaling_fit_fn_an_bn(df_trans, min_len, filter_query, use_pred=False, label=label, enable_constrain=False, fit_variant="r1r3_power")
    
    # plot_an_bn_prediction_errors()
    # scaling_fit_fn_gd(df_trans, 5, f"2e8<N", label=7e9, is_plot=False, prefix="Chinchilla")
    # scaling_fit_torch(df_trans, 5, f"2e8<N", label=7e9, is_plot=False, prefix="Chinchilla_torch")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time:.6f}秒")
