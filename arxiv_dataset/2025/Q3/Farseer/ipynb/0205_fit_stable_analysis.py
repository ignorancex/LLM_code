
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from anyio import current_time
from pandas import DataFrame
from sympy.physics.units import current
from scaling_law_end2end_fit import scaling_fit_fn_an_bn, scaling_fit_fn_gd, scaling_fit_torch
from scalinglaw_utils.scaling_law_fiting.data_filters import filter_data
from scalinglaw_utils.scaling_law_fiting.fitting_utils import cal_mean_error, ax_default_setting
from scalinglaw_utils.scaling_law_fiting.non_linear_model_factory import NonLinearModelFactory
from scalinglaw_utils.scaling_law_fiting.read_data_big_exp import read_data_1222, read_data_bilingual, read_data_code, read_data_val_big_d, read_data_rope, read_data_math, read_data_lr_bs, read_data_1222_smooth_loss, read_data_val_big_d_smooth_loss
from concurrent.futures import ProcessPoolExecutor, as_completed
from scaling_law_analysis_power_law import ax_default_setting, plot_dots, plot_dots_nd
from scalinglaw_utils.scaling_law_fiting.utils_monotonic import check_monotonic_indices

def fit_model(df, max_N, method, write_out):
    print(f'正在拟合 N={max_N}...')
    try:
        if (method == 'self'):
            (res, para) = scaling_fit_fn_an_bn(df, 5, f'2e8<N<{max_N}', False, (- 0.53), False, max_N, write_out, False)
        elif (method == 'torch'):
            (res, para) = scaling_fit_torch(df, 5, f'2e8<N<{max_N}', max_N, write_out, False, prefix='Chinchilla_torch')
        else:
            (res, para) = scaling_fit_fn_gd(df, 5, f'2e8<N<{max_N}', max_N, write_out, False)
        return para
    except Exception as e:
        print(f'拟合 N={max_N} 失败: {e}')
        raise

def fit_model_dn_ratio(df, ratio_min, ratio_max, method, write_out):
    '\n    针对特定D/N比率范围的数据进行拟合\n    \n    Args:\n        df (DataFrame): 输入数据框\n        ratio_min (float): D/N比率范围的最小值\n        ratio_max (float): D/N比率范围的最大值\n        method (str): 拟合方法\n        write_out (bool): 是否将结果写入文件\n    \n    Returns:\n        DataFrame: 拟合参数结果\n    '
    print(f'正在拟合 D/N ratio 范围: {ratio_min:.4f} - {ratio_max:.4f}...')
    try:
        query_str = f'{ratio_min}<=(D/N)<={ratio_max}'
        if (method == 'self'):
            (res, para) = scaling_fit_fn_an_bn(df, 5, query_str, False, (- 0.53), False, None, write_out, False)
        elif (method == 'torch'):
            (res, para) = scaling_fit_torch(df, 5, query_str, None, write_out, False, prefix='Chinchilla_torch')
        else:
            (res, para) = scaling_fit_fn_gd(df, 5, query_str, None, write_out, False)
        para['DN_ratio_min'] = ratio_min
        para['DN_ratio_max'] = ratio_max
        para['DN_ratio_range'] = f'{ratio_min:.4f}_{ratio_max:.4f}'
        return para
    except Exception as e:
        print(f'拟合 D/N ratio 范围 {ratio_min:.4f} - {ratio_max:.4f} 失败: {e}')
        raise

def gen_model_var_N(df: DataFrame, front_num=10, write_out=True, prefix='res_all', method='self', max_workers=20):
    current_time_str = __import__('datetime').datetime.now().strftime('%y%m%d%H%M')
    unique_N = sorted(df['N'].unique(), reverse=True)
    if (len(unique_N) < front_num):
        raise ValueError(f'数据中只有 {len(unique_N)} 个唯一 N 值，不足以进行 {front_num} 次拟合。')
    para_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fit_model, df, unique_N[i], method, write_out) for i in range(front_num)]
        for future in as_completed(futures):
            try:
                para = future.result()
                para_list.append(para)
            except Exception as e:
                print(f'Error occurred: {e}')
    if (not para_list):
        raise ValueError('所有模型拟合失败，无结果可合并。')
    para_all = pd.concat(para_list, axis=0, ignore_index=True)
    para_all.reset_index(drop=True, inplace=True)
    save_name = f'./model/{prefix}_{current_time_str}.csv'
    para_all.sort_values(by=['fit_query', 'Pred_name'], inplace=True)
    para_all.to_csv(save_name, index=False)
    return (para_all, save_name)

def fit_stable_analysis(df: DataFrame, val_dfs: list, val_names: list, ref_mode: DataFrame=None, fit_name: str='Pred_name', fit_version: str='label', fit_class: str='fit_class', prefix=''):
    '\n    :param df: 包含模型参数的 DataFrame\n    :param val_dfs: 包含验证集的 DataFrame 列表\n    :param val_names: 验证集的名称列表，用于标识每个验证集\n    :param ref_mode: 参考模式 DataFrame\n    :param fit_name: 模型名称列名\n    :param fit_version: 模型版本列名\n    :param fit_class: 模型类别列名\n    :param prefix: 输出文件前缀\n    :return: 包含结果的 DataFrame 和输出文件名\n    '
    para_all = df.copy()
    current_time_str = __import__('datetime').datetime.now().strftime('%y%m%d%H%M')
    para_all['UniqueKey'] = ((para_all[fit_name].astype(str) + '_') + para_all[fit_version].astype(str))
    factory = NonLinearModelFactory()
    fits = factory.create_instances_from_dataframe(para_all, fit_class, 'UniqueKey')
    for (idx, row) in para_all.iterrows():
        key = row['UniqueKey']
        fit = fits[key]
        for (val_df, val_name) in zip(val_dfs, val_names):
            df_pred = fit.batch_pred(val_df, 'N', 'D', (key + '_pred'), True)
            (abs_err, rel_err) = cal_mean_error(df_pred, 'L', (key + '_pred'))
            para_all.at[(idx, f'{val_name}_AbsError')] = abs_err
            para_all.at[(idx, f'{val_name}_RelError')] = rel_err
    if (ref_mode is not None):
        unique_refs = ref_mode.drop_duplicates(subset=['Model'])
        for ref in unique_refs['Model']:
            para_all[f'{ref}_pred'] = None
            para_all[f'{ref}_RelError'] = None
        for (idx, model_row) in para_all.iterrows():
            key = model_row['UniqueKey']
            fit = fits[key]
            for (_, ref_row) in unique_refs.iterrows():
                ref_name = ref_row['Model']
                N_ref = ref_row['N']
                D_ref = ref_row['D']
                true_bpw = ref_row['BPW']
                pred_val = fit.predict(N_ref, D_ref)
                rel_err = ((pred_val - true_bpw) / true_bpw)
                para_all.at[(idx, f'{ref_name}_pred')] = pred_val
                para_all.at[(idx, f'{ref_name}_RelError')] = rel_err
    output_filename = f'./model/{prefix}_res_all_{current_time_str}.csv'
    para_all.to_csv(output_filename, index=False)
    print(f'结果已保存到 {output_filename}')
    return (para_all, output_filename)

def plot_error_variation(para_all, val_names=['Val1_RelError', 'Val2_RelError'], fit_name='Pred_name', fit_version='label', tail_n=0, show=True, prefix=''):
    '\n    根据 para_all 数据，绘制 12 行×16 列的子图，每个子图展示一个模型（Pred_name）在一种相对误差下\n    随着 label (max_N) 的变化趋势。每个子图标题为 \'Pred_name + 误差类型\'，横轴为 label，\n    纵轴为相对误差。\n    参数:\n      para_all: 包含模型参数和误差数据的 DataFrame，\n                必须包含 fit_name 列（例如 "Pred_name"）、fit_version 列（例如 "label"），\n                以及若干个以 "RelError" 结尾的列（共12个）。\n      tailn   : 每个子图绘制最后 tailn 个数据点，默认值为 5。\n      fit_name: 模型名称列名，默认 "Pred_name"\n      fit_version: 标签列名，默认 "label"\n    '
    models = sorted(para_all[fit_name].unique())
    base_errors = [(s + '_RelError') for s in val_names]
    other_errors = [col for col in para_all.columns if (col.endswith('RelError') and (col not in base_errors))]
    rel_error_cols = (base_errors + sorted(other_errors))
    (n_rows, n_cols) = (len(rel_error_cols), len(models))
    (fig, axes) = plt.subplots(n_rows, n_cols, figsize=((n_cols * 4), (n_rows * 3)), sharex=True, sharey=False, dpi=300)
    if ((n_rows == 1) or (n_cols == 1)):
        axes = axes.reshape(n_rows, n_cols)
    for (i, err) in enumerate(rel_error_cols):
        for (j, model) in enumerate(models):
            ax = axes[(i, j)]
            d = para_all[(para_all[fit_name] == model)].sort_values(fit_version)
            if (tail_n > 0):
                d = d.tail(tail_n)
            sns.lineplot(x=d[fit_version], y=d[err], marker='.', ax=ax)
            ax.axhline(y=0, color='r', linestyle='--')
            ax_default_setting(ax)
            if (not d.empty):
                last_val = d[err].iloc[(- 1)]
                title_str = f'{model} + {err} (last: {last_val:.5f})'
            else:
                title_str = f'{model} + {err}'
            ax.set_title(title_str, fontsize=8)
            ax.tick_params(axis='y', labelleft=True)
    plt.tight_layout()
    if show:
        plt.show()
    current_time_str = __import__('datetime').datetime.now().strftime('%y%m%d%H%M')
    plt.savefig(f'./figures/{prefix}_error_variation_{current_time_str}.png', dpi=300)
    plt.savefig(f'./figures/{prefix}_error_variation_{current_time_str}.pdf')

def plot_error_variation_single_version(para_all, df_val, fit_name='Pred_name', fit_class='fit_class', gt_name='L', X_D='D', X_N='N', group_name='N'):
    val_group = sorted(df_val[group_name].unique())
    fits = sorted(para_all[fit_name].unique())
    (n_rows, n_cols) = (len(val_group), (len(fits) + 2))
    print(n_rows, n_cols)
    (fig, axes) = plt.subplots(n_rows, n_cols, figsize=((n_cols * 5), (n_rows * 4)), dpi=300)
    factory = NonLinearModelFactory()
    fits = factory.create_instances_from_dataframe(para_all, fit_class, fit_name)
    for (i, (key, fit)) in enumerate(fits.items()):
        df_pred = fit.batch_pred(df_val, X_N, X_D, f'{gt_name}_pred', True)
        df_pred[f'{gt_name}_error'] = (df_pred[gt_name] - df_pred[f'{gt_name}_pred'])
        df_pred[f'{gt_name}_r_error'] = (df_pred[f'{gt_name}_error'] / df_pred[gt_name])
        for (j, val) in enumerate(val_group):
            df_pred_j = df_pred.query(f'{group_name} == {val}')
            abs_mean_error = df_pred_j[f'{gt_name}_error'].abs().mean()
            abs_mean_r_error = df_pred_j[f'{gt_name}_r_error'].abs().mean()
            sns.lineplot(data=df_pred_j, x=X_D, y=gt_name, ax=axes[j][i], label=f'GT:{gt_name}')
            sns.lineplot(data=df_pred_j, x=X_D, y=f'{gt_name}_pred', ax=axes[j][i], label=f'{key} Pred')
            sns.lineplot(data=df_pred_j, x=X_D, y=f'{gt_name}_error', ax=axes[j][(n_cols - 2)], label=f'{key}_error={abs_mean_error:.3e}')
            sns.lineplot(data=df_pred_j, x=X_D, y=f'{gt_name}_r_error', ax=axes[j][(n_cols - 1)], label=f'{key}_r_error={abs_mean_r_error:.3e}')
            ax_default_setting([axes[j][i], axes[j][(n_cols - 2)], axes[j][(n_cols - 1)]])
            axes[j][i].set_title(f'{X_N}={val}')
            axes[j][(n_cols - 1)].set_title(f'{X_N}={val}')
            axes[j][(n_cols - 2)].set_title(f'{X_N}={val}')
    plt.tight_layout()
    plt.show()

def compute_metric_for_error(para_all, tailn, fit_name, fit_version, error_col, metric_func):
    '\n    对于每个模型，提取该模型下按照 fit_version 排序后尾部 tailn 个数据点（针对 error_col 列），\n    并计算指标值（metric_func 接收一个 Series）。\n    返回 DataFrame，包含三列：model、value（计算得到的指标值）、last_label（尾部数据中最后一条的 label）。\n    '
    models = sorted(para_all[fit_name].unique())
    results = []
    for model in models:
        d = para_all[(para_all[fit_name] == model)].sort_values(fit_version).tail(tailn)
        if d.empty:
            results.append((model, np.nan, np.nan))
        else:
            value = metric_func(d[error_col])
            last_label = d[fit_version].iloc[(- 1)]
            results.append((model, value, last_label))
    return pd.DataFrame(results, columns=['model', 'value', 'last_label'])

def plot_metric_figure(para_all, tailn, fit_name='Pred_name', fit_version='label', rel_error_cols=None, metric_func=None, metric_name='Metric'):
    '\n    绘制一个图，图中包含 12 个子图（3 行×4 列），每个子图对应一个相对误差列。\n    对于每个相对误差列，从 para_all 中按模型提取尾部 tailn 个数据（按 fit_version 排序），\n    用 metric_func 计算指标，结果以柱状图展示，x 轴为模型，y 轴为计算指标的值。\n\n    每个子图标题中附上该相对误差列下"尾部数据"中最大 fit_version 对应的指标值（保留 5 位小数）。\n    '
    if (rel_error_cols is None):
        base_errors = ['Val1_RelError', 'Val2_RelError']
        other_errors = [col for col in para_all.columns if (col.endswith('RelError') and (col not in base_errors))]
        rel_error_cols = (base_errors + sorted(other_errors))
    (fig, axes) = plt.subplots(3, 4, figsize=(24, 12), dpi=300)
    axes = axes.flatten()
    for (i, error_col) in enumerate(rel_error_cols):
        ax = axes[i]
        df_metric = compute_metric_for_error(para_all, tailn, fit_name, fit_version, error_col, metric_func)
        x = np.arange(len(df_metric))
        ax.bar(x, df_metric['value'], color='skyblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(df_metric['model'], rotation=45, fontsize=8)
        ax.set_title(f'{error_col},{metric_name}', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
    for j in range((i + 1), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_three_metrics(paras_and_fit, tail_n):
    plot_metric_figure(paras_and_fit, tailn=tail_n, fit_name='Pred_name', fit_version='label', metric_func=(lambda s: s.var()), metric_name='Variance')
    plot_metric_figure(paras_and_fit, tailn=tail_n, fit_name='Pred_name', fit_version='label', metric_func=(lambda s: s.iloc[(- 1)]), metric_name='Last Value')
    plot_metric_figure(paras_and_fit, tailn=(tail_n - 1), fit_name='Pred_name', fit_version='label', metric_func=(lambda s: (np.mean(np.diff(s)) if (len(s) >= 2) else np.nan)), metric_name='Avg Diff')

def _agg_tail_metric(para_all, group_col, tailn, fit_version, value_col, metric_func):
    "\n    对 para_all 按 group_col 分组（可能是模型或误差列），\n    每组先按 fit_version 排序，再取后 tailn 个数据，\n    最后在 value_col 列上应用 metric_func 计算指标。\n\n    返回 DataFrame，包含 [group_col, 'MetricValue'] 两列。\n    "
    groups = sorted(para_all[group_col].unique())
    rows = []
    for g in groups:
        sub = para_all[(para_all[group_col] == g)].sort_values(fit_version).tail(tailn)
        if sub.empty:
            val = np.nan
        else:
            val = metric_func(sub[value_col])
        rows.append((g, val))
    return pd.DataFrame(rows, columns=[group_col, 'MetricValue'])

def _plot_subplots(df_metric, group_col, nrows, ncols, title_prefix):
    '\n    将 df_metric 按 nrows×ncols 的网格绘制子图。\n    df_metric 必须包含三列：\n      - group_col：决定子图分组（例如误差列或模型），\n      - x_label：横轴刻度（例如模型名称或误差列名称），\n      - MetricValue：计算得到的指标值。\n\n    每个子图的标题格式为：\n         "{title_prefix}: {组名}\nMean: {均值}  Max: {最大值}"\n    其中组名和 x 轴刻度中的 "RelError" 字符串会被去掉。\n    同时每个子图增加网格。\n    '
    (fig, axes) = plt.subplots(nrows, ncols, figsize=((ncols * 6), (nrows * 3)), dpi=300)
    axes = axes.flatten()
    unique_items = sorted(df_metric[group_col].unique())
    for (i, item) in enumerate(unique_items):
        ax = axes[i]
        sub = df_metric[(df_metric[group_col] == item)]
        x = np.arange(len(sub))
        ax.bar(x, sub['MetricValue'], color='skyblue', edgecolor='black')
        x_labels = [str(lbl).replace('RelError', '') for lbl in sub['x_label']]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
        m_val = sub['MetricValue'].mean()
        abs_m_val = sub['MetricValue'].abs().mean()
        max_val = sub['MetricValue'].max()
        item_clean = str(item).replace('_RelError', '')
        item_clean = str(item_clean).replace('LLaMA', '')
        ax.set_title(f'''{title_prefix}: {item_clean}
Mean: {m_val:.6f} abs Mean:{abs_m_val:.6f} Max: {max_val:.5f}''', fontsize=9)
        ax.grid(True)
    for j in range((i + 1), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_by_rel_errors(para_all, tailn=5, fit_name='Pred_name', fit_version='label'):
    '\n    绘制3张图（每张对应一个指标），每张图各包含12个子图（3×4布局）。\n    每个子图对应一个"相对误差列"（共12个），横轴显示不同模型（fit_name）。\n\n    指标包括：\n      - "Variance"：尾部数据的方差（这里采用 var()）；\n      - "LastValue"：尾部数据的最后一个值；\n      - "MeanDiff"：尾部数据中相邻数据的差值均值（当 tailn<2 时返回 NaN）。\n    '
    base_errors = ['Val1_RelError', 'Val2_RelError']
    other_errors = [c for c in para_all.columns if (c.endswith('RelError') and (c not in base_errors))]
    rel_error_cols = (base_errors + sorted(other_errors))
    metrics = {'Variance': (lambda s: s.var()), 'LastValue': (lambda s: (s.iloc[(- 1)] if (len(s) > 0) else np.nan)), 'MeanDiff': (lambda s: (np.mean(np.diff(s)) if (len(s) > 1) else np.nan))}
    for (metric_name, metric_func) in metrics.items():
        rows = []
        for err_col in rel_error_cols:
            df_agg = _agg_tail_metric(para_all, fit_name, tailn, fit_version, err_col, metric_func)
            for (_, r) in df_agg.iterrows():
                rows.append({'group_col': err_col, 'x_label': r[fit_name], 'MetricValue': r['MetricValue']})
        df_metric = pd.DataFrame(rows)
        print(f'Plotting by RelErrors => {metric_name}')
        _plot_subplots(df_metric, 'group_col', 3, 4, title_prefix=metric_name)

def plot_by_models(para_all, tailn=5, fit_name='Pred_name', fit_version='label'):
    '\n    绘制3张图（每张对应一个指标），每张图各包含16个子图（5×4布局，其中多余的子图会被隐藏）。\n    每个子图对应一个模型（共16个），横轴显示不同的相对误差列（共12个）。\n\n    指标包括：\n      - "Variance"：尾部数据的方差（var）；\n      - "LastValue"：尾部数据的最后一个值；\n      - "MeanDiff"：尾部数据中相邻数据的差值均值（当 tailn<2 时返回 NaN）。\n    '
    base_errors = ['Val1_RelError', 'Val2_RelError']
    other_errors = [c for c in para_all.columns if (c.endswith('RelError') and (c not in base_errors))]
    rel_error_cols = (base_errors + sorted(other_errors))
    models = sorted(para_all[fit_name].unique())
    metrics = {'StdDev': (lambda s: s.var()), 'LastValue': (lambda s: (s.iloc[(- 1)] if (len(s) > 0) else np.nan)), 'MeanDiff': (lambda s: (np.mean(np.diff(s)) if (len(s) > 1) else np.nan))}
    for (metric_name, metric_func) in metrics.items():
        rows = []
        for model in models:
            for err_col in rel_error_cols:
                sub = para_all[(para_all[fit_name] == model)].sort_values(fit_version).tail(tailn)
                vals = sub[err_col]
                val_metric = (metric_func(vals) if (len(vals) > 0) else np.nan)
                rows.append({'group_col': model, 'x_label': err_col, 'MetricValue': val_metric})
        df_metric = pd.DataFrame(rows)
        print(f'Plotting by Models => {metric_name}')
        _plot_subplots(df_metric, 'group_col', 5, 4, title_prefix=metric_name)

def monotonic_fit_test():
    save_file_1 = '../ipynb/model/fit_250429184008.csv'
    paras = pd.read_csv(save_file_1)
    factory = NonLinearModelFactory()
    fits = factory.create_instances_from_dataframe(paras, 'fit_class', 'Pred_name', decimal_mode=False)
    datas = np.concatenate([np.arange(10000000000.0, 1000000000000.0, 1000000000.0), np.arange(1000000000000.0, 100000000000000.0, 100000000000.0), np.arange(100000000000000.0, 1e+16, 10000000000000.0)])
    no_mono_name = set()
    for (k, (name, fit)) in enumerate(fits.items()):
        print(((('\\text{' + name.replace('_', '-')) + '}') + f'\\ {fit} \\'))
    for data in datas.tolist():
        print(f'Checking data={data}')
        model_sizes = np.arange((data / 500), (data * 10), (data / 500))
        data_array = np.full_like(model_sizes, data)
        no_mono_name_sub = set()
        for (k, (name, fit)) in enumerate(fits.items()):
            losses = fit.pred_ndarray(model_sizes, data_array)
            is_mono = check_monotonic_indices(losses, 'descending', True)
            if (len(is_mono) > 0):
                no_mono_name.add(name)
                no_mono_name_sub.add(name)
        print(f'no monotonic in data {data} is {no_mono_name_sub}')
    print(f'no monotonic: {no_mono_name}')

def unit_test_pred_with_given_N():
    save_file_1 = '../ipynb/model/fit_250429184008.csv'
    paras = pd.read_csv(save_file_1)
    factory = NonLinearModelFactory()
    fits = factory.create_instances_from_dataframe(paras, 'fit_class', 'Pred_name', decimal_mode=True)
    (model_size_1, model_size_2) = (670000000000.0, 300000000000.0)
    tokens = np.arange(1000000000000.0, 3000000000000.0, 1000000000000.0)
    print(tokens)
    print(fits['L_pred3_c'].predict(model_size_1, 1000000000000.0), fits['L_pred3_c'].predict(model_size_1, 2000000000000.0))
    print(fits['L_pred3_c'].pred_with_given_N(model_size_1, tokens))
    for (k, (name, fit)) in enumerate(fits.items()):
        loss1 = fit.predict(model_size_1, 10000000000000.0)
        loss2 = fit.predict(model_size_2, 10000000000000.0)
        print(f'{name},loss1:{loss1},loss2:{loss2}')

def gen_model_var_DN_ratio(df: DataFrame, ratio_bins=10, write_out=True, prefix='res_all_dn_ratio', method='self', max_workers=20):
    '\n    基于D/N比率对数据进行分组分析\n    \n    Args:\n        df (DataFrame): 输入数据框\n        ratio_bins (int): D/N比率分组的数量\n        write_out (bool): 是否将结果写入文件\n        prefix (str): 输出文件名前缀\n        method (str): 拟合方法\n        max_workers (int): 并行处理的最大工作进程数\n    \n    Returns:\n        Tuple[DataFrame, str]: 返回处理后的参数DataFrame和保存的文件路径\n    '
    current_time_str = __import__('datetime').datetime.now().strftime('%y%m%d%H%M')
    df['DN_ratio'] = round((df['D'] / df['N']), 1)
    unique_ratios = df['DN_ratio'].unique()
    min_ratio = np.min(unique_ratios)
    max_ratio = np.max(unique_ratios)
    ratio_ranges = np.exp(np.linspace(np.log(min_ratio), np.log(max_ratio), (ratio_bins + 1)))
    para_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(ratio_bins):
            ratio_min = ratio_ranges[0]
            ratio_max = ratio_ranges[(i + 1)]
            if (len(df) > 0):
                futures.append(executor.submit(fit_model_dn_ratio, df, ratio_min, ratio_max, method, write_out))
        for future in as_completed(futures):
            try:
                para = future.result()
                if (para is not None):
                    para_list.append(para)
            except Exception as e:
                print(f'Error occurred: {e}')
    if (not para_list):
        raise ValueError('所有模型拟合失败，无结果可合并。')
    para_all = pd.concat(para_list, axis=0, ignore_index=True)
    para_all.reset_index(drop=True, inplace=True)
    save_name = f'./model/{prefix}_{current_time_str}.csv'
    para_all.sort_values(by=['fit_query', 'Pred_name'], inplace=True)
    para_all.to_csv(save_name, index=False)
    return (para_all, save_name)

def fit_stable_analysis_dn_ratio(df: DataFrame, val_dfs: list, val_names: list, ref_mode: DataFrame=None, fit_name: str='Pred_name', prefix=''):
    '\n    针对DN比率分组的稳定性分析函数\n    \n    Args:\n        df: 包含模型参数的 DataFrame，必须包含 DN_ratio_range 列\n        val_dfs: 包含验证集的 DataFrame 列表\n        val_names: 验证集的名称列表\n        ref_mode: 参考模式 DataFrame\n        fit_name: 模型名称列名\n        prefix: 输出文件前缀\n    '
    para_all = df.copy()
    current_time_str = __import__('datetime').datetime.now().strftime('%y%m%d%H%M')
    para_all['UniqueKey'] = ((para_all[fit_name].astype(str) + '_') + para_all['DN_ratio_range'].astype(str))
    factory = NonLinearModelFactory()
    fits = factory.create_instances_from_dataframe(para_all, 'fit_class', 'UniqueKey')
    for (idx, row) in para_all.iterrows():
        key = row['UniqueKey']
        fit = fits[key]
        dn_range = row['DN_ratio_range']
        for (val_df, val_name) in zip(val_dfs, val_names):
            df_pred = fit.batch_pred(val_df, 'N', 'D', (key + '_pred'), True)
            (abs_err, rel_err) = cal_mean_error(df_pred, 'L', (key + '_pred'))
            para_all.at[(idx, f'{val_name}_AbsError')] = abs_err
            para_all.at[(idx, f'{val_name}_RelError')] = rel_err
    if (ref_mode is not None):
        unique_refs = ref_mode.drop_duplicates(subset=['Model'])
        for ref in unique_refs['Model']:
            para_all[f'{ref}_pred'] = None
            para_all[f'{ref}_RelError'] = None
        for (idx, model_row) in para_all.iterrows():
            key = model_row['UniqueKey']
            fit = fits[key]
            for (_, ref_row) in unique_refs.iterrows():
                ref_name = ref_row['Model']
                N_ref = ref_row['N']
                D_ref = ref_row['D']
                true_bpw = ref_row['BPW']
                pred_val = fit.predict(N_ref, D_ref)
                rel_err = ((pred_val - true_bpw) / true_bpw)
                para_all.at[(idx, f'{ref_name}_pred')] = pred_val
                para_all.at[(idx, f'{ref_name}_RelError')] = rel_err
    output_filename = f'./model/{prefix}_dn_ratio_analysis_{current_time_str}.csv'
    para_all.to_csv(output_filename, index=False)
    print(f'结果已保存到 {output_filename}')
    return (para_all, output_filename)

def plot_error_variation_dn_ratio(para_all, val_names=['Val1_RelError', 'Val2_RelError'], fit_name='Pred_name', show=True, prefix=''):
    '\n    绘制基于DN比分析的误差变化图。每个子图展示一个模型在不同DN比率范围下的相对误差变化。\n    \n    参数:\n      para_all: 包含模型参数和误差数据的 DataFrame，\n                必须包含 fit_name 列（例如 "Pred_name"）、DN_ratio_range 列，\n                以及若干个以 "RelError" 结尾的列。\n      val_names: 验证集名称列表\n      fit_name: 模型名称列名，默认 "Pred_name"\n      show: 是否显示图形\n      prefix: 输出文件名前缀\n    '
    models = sorted(para_all[fit_name].unique())
    base_errors = [(s + '_RelError') for s in val_names]
    other_errors = [col for col in para_all.columns if (col.endswith('RelError') and (col not in base_errors))]
    rel_error_cols = (base_errors + sorted(other_errors))
    (n_rows, n_cols) = (len(rel_error_cols), len(models))
    (fig, axes) = plt.subplots(n_rows, n_cols, figsize=((n_cols * 4), (n_rows * 3)), sharex=True, sharey=False, dpi=300)
    if ((n_rows == 1) or (n_cols == 1)):
        axes = axes.reshape(n_rows, n_cols)
    for (i, err) in enumerate(rel_error_cols):
        for (j, model) in enumerate(models):
            ax = axes[(i, j)]
            d = para_all[(para_all[fit_name] == model)].copy()
            d['sort_key'] = d['DN_ratio_range'].apply((lambda x: float(x.split('_')[1])))
            d = d.sort_values('sort_key')
            sns.lineplot(x=d['DN_ratio_range'], y=d[err], marker='.', ax=ax)
            ax.axhline(y=0, color='r', linestyle='--')
            ax_default_setting(ax)
            if (not d.empty):
                mean_err = d[err].mean()
                std_err = d[err].std()
                min_err = d[err].min()
                max_err = d[err].max()
                title_str = f'''{model}
{err.replace('_RelError', '')}
mean={mean_err:.2e}, std={std_err:.2e}
min={min_err:.2e}, max={max_err:.2e}'''
            else:
                title_str = f'''{model}
{err}'''
            ax.set_title(title_str, fontsize=8)
            ax.tick_params(axis='y', labelleft=True)
            if (i == (n_rows - 1)):
                ax.set_xlabel('D/N Ratio Range')
            if (j == 0):
                ax.set_ylabel('Relative Error')
            ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if show:
        plt.show()
    current_time_str = __import__('datetime').datetime.now().strftime('%y%m%d%H%M')
    plt.savefig(f'./figures/{prefix}_error_variation_dn_{current_time_str}.png', dpi=300)
    plt.savefig(f'./figures/{prefix}_error_variation_dn_{current_time_str}.pdf')
if (__name__ == '__main__'):
    df_code = read_data_code()
    df_trans = filter_data(df_code, min_D=1)
    print(df_trans.shape)
    (para_all_1, save_file_1) = gen_model_var_N(df_trans, front_num=7, method='self')
    df_val = df_trans.query('N>3e9')
    df_val_6b = df_val.query('D>6e9')
    df_val_1 = df_val.query('6e9<D<4e10')
    df_val_2 = df_val.query('D>4e10')
    df_val_more = df_trans.query('N>3e9')
    df_val_big_D = read_data_val_big_d()
    df_val_big_D_2b = df_val_big_D.query('N<3e9')
    df_val_big_D_all = df_val_big_D.query('N<25e9')
    df_val_big_D_3b = df_val_big_D.query('6e9>N>3e9')
    df_val_big_D_6b = df_val_big_D.query('7e9>N>6e9')
    df_val_big_D_25b = df_val_big_D.query('N>25e9')
    df_val_middle_DN_two = df_val_big_D.query('10<`round(D/N)`<85')
    llama_model = pd.read_csv('./model/llama.csv')
    df_val_code = pd.read_csv('data/val_code.csv', dtype={'N': float, 'D': float})
    df_val_code.rename(columns={'IntelliValSet_code': 'L'}, inplace=True)
    llama_model = llama_model.query('BPW>0')
    paras = pd.read_csv(save_file_1)
    val_dfs = [df_val_6b, df_val_1, df_val_2, df_val_more, df_val_big_D_2b, df_val_big_D_3b, df_val_big_D_6b, df_val_big_D_25b, df_val_middle_DN_two, df_val_big_D_all]
    val_names = ['Val_6b', 'Val1', 'Val2', 'ValMore', 'ValBigN2b', 'ValBigN3b', 'ValBigN6b', 'ValBigN25b', 'ValMiddleDN2', 'ValBigD']
    prefix = 'code_review'
    (para_all_2, save_file_2) = fit_stable_analysis(paras, val_dfs, val_names, llama_model, prefix=prefix)
    paras_and_fit = pd.read_csv(save_file_2)
    plot_error_variation(paras_and_fit, val_names, fit_name='Pred_name', show=False, prefix=prefix)
