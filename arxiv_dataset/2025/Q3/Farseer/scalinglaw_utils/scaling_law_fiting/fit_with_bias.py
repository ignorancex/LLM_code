
import pandas as pd
import numpy as np
from pandas import DataFrame
from tabulate import tabulate
from scalinglaw_utils.basic_fitting_tool import LinearFit
import seaborn as sns
import matplotlib.pyplot as plt
from scalinglaw_utils.scaling_law_fiting.fitting_utils import ax_default_setting

def fit_with_y_bias(df: DataFrame, x_key: str, y_key: str, bias: float=0, x_log=True, y_log=True, weight=None, suffix=''):
    df_c = df.copy().sort_values([x_key, y_key])
    residuals_df = pd.DataFrame(index=df_c.index)
    res = list()
    prefix = (((('x-' + x_key) + '_y-') + y_key) + suffix)
    (X, Y_RAW) = (df_c[x_key].to_numpy(), df_c[y_key].to_numpy())
    Y = (Y_RAW - bias)
    weights = (df_c[weight].values if (weight is not None) else None)
    model = LinearFit(X, Y).fit(x_log, y_log, 'sklearn', weights)
    residuals_df.loc[(df_c.index, (prefix + '_pred_raw'))] = (model.pred_raw + bias)
    residuals_df.loc[(df_c.index, (prefix + '_pred_y'))] = model.pred_y
    residuals_df.loc[(df_c.index, (prefix + '_residual'))] = model.residual
    residuals_df.loc[(df_c.index, (prefix + '_relative_residual'))] = (model.residual / (Y_RAW + 1e-10))
    residuals_df.loc[(df_c.index, (prefix + '_slope'))] = model.slope
    residuals_df.loc[(df_c.index, (prefix + '_intercept'))] = model.intercept
    fit_res = [prefix, bias, model.slope, model.intercept, model.residual_mean, np.mean(abs((model.residual / (Y_RAW + 1e-10)))), model.residual_trans_mean, abs(model.slope)]
    res.append(fit_res)
    df_c = pd.concat([df_c, residuals_df], axis=1)
    columns_name = ['name', 'bias', 'slope', 'intercept', 'residual_mean', 'relative_residual_mean', 'residual_trans_mean', 'slope_abs']
    res_df = pd.DataFrame(res, columns=columns_name)
    return (df_c, res_df)

def found_best_bias(df: DataFrame, x_key: str, y_key: str, bias: list[float], x_log=True, y_log=True, weight=None, is_plot=True, extra_plot: list[list]=None, is_print=True):
    (start, end, step) = (bias[0], bias[1], bias[2])
    assert ((start < end) and (step < (end - start)))
    bias_values = np.arange(start, end, step)
    (df_fit_list, fit_res_list) = ([], [])
    (MIN_relative_residual, MIN_index, MIN_bias) = (float('inf'), 0, float('inf'))
    for i in range(len(bias_values)):
        (df_fit, fit_res) = fit_with_y_bias(df, x_key, y_key, bias=float(bias_values[i]), x_log=x_log, y_log=y_log, weight=weight)
        df_fit_list.append(df_fit)
        fit_res_list.append(fit_res)
        if (fit_res['relative_residual_mean'].item() < MIN_relative_residual):
            MIN_relative_residual = fit_res['relative_residual_mean'].item()
            (MIN_index, MIN_bias) = (i, bias_values[i])
    fits_res_df = pd.concat(fit_res_list, ignore_index=True)
    if is_plot:
        sns.set_theme(style='whitegrid')
        cols = (5 if (extra_plot is not None) else 4)
        (fig, axs) = plt.subplots(2, cols, figsize=((cols * 4), 10), dpi=300)
        names = ['slope', 'intercept', 'residual_mean', 'relative_residual_mean']
        for j in range(len(names)):
            sns.lineplot(ax=axs[(1, j)], data=fits_res_df, x='bias', y=names[j], label=names[j])
            ax_default_setting(axs[1][j])
        (df_fit_wo_bias, fit_res_wo_bias) = fit_with_y_bias(df, x_key, y_key, bias=0, x_log=x_log, y_log=y_log, weight=weight)
        prefix = ((('x-' + x_key) + '_y-') + y_key)
        suffix = ['_pred_raw', '_pred_raw', '_residual', '_relative_residual']
        for j in range(2):
            sns.lineplot(ax=axs[0][j], data=df_fit_wo_bias, x=x_key, y=y_key, label=f'raw', linestyle='--', marker='o')
        for j in range(len(suffix)):
            sns.lineplot(ax=axs[0][j], data=df_fit_wo_bias, x=x_key, y=(prefix + suffix[j]), label=('pred' + ('' if (j < 2) else f"={fit_res_wo_bias[(suffix[j][1:] + '_mean')].item()}")))
            sns.lineplot(ax=axs[0][j], data=df_fit_list[MIN_index], x=x_key, y=(prefix + suffix[j]), label=(f'bias_{MIN_bias}' + ('' if (j < 2) else f"={fit_res_list[MIN_index][(suffix[j][1:] + '_mean')].item()}")))
            ax_default_setting(axs[0][j])
        if (extra_plot is not None):
            sns.lineplot(x=bias_values, y=extra_plot[0], ax=axs[0][4], label='extra_0')
            sns.lineplot(x=bias_values, y=extra_plot[1], ax=axs[1][4], label='extra_1')
            ax_default_setting([axs[0][4], axs[1][4]])
        axs[0][1].set_xscale('log')
        axs[0][1].set_yscale('log')
        plt.tight_layout()
        plt.show()
    elif is_print:
        print(tabulate(fits_res_df, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.6f'))
    return (MIN_bias, MIN_relative_residual)
