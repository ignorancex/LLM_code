
import pandas as pd
import numpy as np
from pandas import DataFrame
from tabulate import tabulate
from scalinglaw_utils.basic_fitting_tool import LinearFit
import seaborn as sns
import matplotlib.pyplot as plt
from scalinglaw_utils.scaling_law_fiting.fitting_utils import ax_default_setting

def basic_fit(df_c, X, Y, x_log, y_log, prefix, weights):
    residuals_df = pd.DataFrame(index=df_c.index)
    model = LinearFit(X, Y).fit(x_log, y_log, 'sklearn', weights)
    residuals_df.loc[(df_c.index, (prefix + '_pred_raw'))] = model.pred_raw
    residuals_df.loc[(df_c.index, (prefix + '_pred_y'))] = model.pred_y
    residuals_df.loc[(df_c.index, (prefix + '_residual'))] = model.residual
    residuals_df.loc[(df_c.index, (prefix + '_relative_residual'))] = model.relative_residual
    residuals_df.loc[(df_c.index, (prefix + '_slope'))] = model.slope
    residuals_df.loc[(df_c.index, (prefix + '_intercept'))] = model.intercept
    fit_res = [model.slope, model.intercept, model.residual_mean, model.relative_residual_mean, model.residual_trans_mean, abs(model.slope)]
    df_c = pd.concat([df_c, residuals_df], axis=1)
    return (df_c, fit_res)

def fit_with_x_trans(df: DataFrame, x_key: str, y_key: str, factor: float=1, x_log=False, y_log=True, weight=None):
    df_c = df.copy().sort_values([x_key, y_key])
    df_c[f'{x_key}_trans'] = (df_c[x_key] ** factor)
    res = list()
    prefix = ((('x-' + x_key) + '_y-') + y_key)
    (X, Y) = (df_c[f'{x_key}_trans'].to_numpy(), df_c[y_key].to_numpy())
    weights = (df_c[weight].values if (weight is not None) else None)
    (df_c, fit_res) = basic_fit(df_c, X, Y, x_log, y_log, prefix, weights)
    fit_res = ([prefix, factor] + fit_res)
    res.append(fit_res)
    columns_name = ['prefix', 'factor', 'slope', 'intercept', 'residual_mean', 'relative_residual_mean', 'residual_trans_mean', 'slope_abs']
    res_df = pd.DataFrame(res, columns=columns_name)
    return (df_c, res_df)

def found_best_factor(df: DataFrame, x_key: str, y_key: str, factor: list[float], x_log=False, y_log=True, weight=None, is_plot=True, extra_plot: list[list]=None, is_print=True):
    (start, end, step) = (factor[0], factor[1], factor[2])
    min_factor_abs = (factor[3] if (len(factor) > 3) else 0)
    assert ((start < end) and (step < (end - start)))
    fact_values = np.arange(start, end, step)
    (df_fit_list, fit_res_list, factor_list) = ([], [], [])
    (MIN_relative_residual, MIN_index, MIN_fact) = (float('inf'), 0, float('inf'))
    for i in range(len(fact_values)):
        if (abs(fact_values[i]) < min_factor_abs):
            continue
        formatted_fact = float('{:.8f}'.format(fact_values[i]))
        (df_fit, fit_res) = fit_with_x_trans(df, x_key, y_key, formatted_fact, x_log=x_log, y_log=y_log, weight=weight)
        df_fit_list.append(df_fit)
        fit_res_list.append(fit_res)
        factor_list.append(fact_values[i])
        if (fit_res['relative_residual_mean'].item() < MIN_relative_residual):
            MIN_relative_residual = fit_res['relative_residual_mean'].item()
            (MIN_index, MIN_fact) = (i, formatted_fact)
    fits_res_df = pd.concat(fit_res_list, ignore_index=True)
    if is_plot:
        sns.set_theme(style='whitegrid')
        cols = (5 if (extra_plot is not None) else 4)
        (fig, axs) = plt.subplots(2, cols, figsize=((cols * 4), 10), dpi=300)
        names = ['slope', 'intercept', 'residual_mean', 'relative_residual_mean']
        for j in range(len(names)):
            sns.lineplot(ax=axs[(1, j)], data=fits_res_df, x='factor', y=names[j], label=names[j])
            ax_default_setting(axs[1][j])
        (df_fit_wo_bias, fit_res_wo_bias) = fit_with_x_trans(df, x_key, y_key, 1, x_log=True, y_log=True, weight=weight)
        prefix = ((('x-' + x_key) + '_y-') + y_key)
        suffix = ['_pred_raw', '_pred_raw', '_residual', '_relative_residual']
        for j in range(2):
            sns.lineplot(ax=axs[0][j], data=df_fit_wo_bias, x=x_key, y=y_key, label=f'raw', linestyle='--', marker='o')
        for j in range(len(suffix)):
            sns.lineplot(ax=axs[0][j], data=df_fit_wo_bias, x=x_key, y=(prefix + suffix[j]), label=('power-law' + ('' if (j < 2) else f"={fit_res_wo_bias[(suffix[j][1:] + '_mean')].item()}")))
            sns.lineplot(ax=axs[0][j], data=df_fit_list[MIN_index], x=x_key, y=(prefix + suffix[j]), label=(f'fact_{MIN_fact}' + ('' if (j < 2) else f"={fit_res_list[MIN_index][(suffix[j][1:] + '_mean')].item()}")))
            ax_default_setting(axs[0][j])
        if (extra_plot is not None):
            sns.lineplot(x=factor_list, y=extra_plot[0], ax=axs[0][4], label='extra_0')
            sns.lineplot(x=factor_list, y=extra_plot[1], ax=axs[1][4], label='extra_1')
            ax_default_setting([axs[0][4], axs[1][4]])
        axs[0][1].set_xscale('log')
        axs[0][1].set_yscale('log')
        plt.tight_layout()
        plt.show()
    elif is_print:
        print(tabulate(fits_res_df, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.6f'))
    return (MIN_fact, MIN_relative_residual)
