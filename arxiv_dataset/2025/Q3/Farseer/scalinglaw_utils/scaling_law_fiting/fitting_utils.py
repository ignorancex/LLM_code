
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

def rename_columns_with_prefix(df, prefix, suffix):
    '\n    为DataFrame中以特定前缀开头的列名添加后缀，并检查新列名是否与现有列名冲突。\n\n    参数:\n    df (pd.DataFrame): 输入的DataFrame。\n    prefix (str): 要匹配的列名前缀。\n    suffix (str): 要添加到列名的后缀。\n\n    返回:\n    pd.DataFrame: 修改后的DataFrame。\n\n    异常:\n    ValueError: 如果新列名与现有列名冲突。\n    '
    cols_to_rename = df.filter(regex=f'^{prefix}').columns
    new_col_names = [(col + suffix) for col in cols_to_rename]
    if (set(new_col_names) & set(df.columns)):
        raise ValueError('新列名与现有列名冲突')
    col_mapping = dict(zip(cols_to_rename, new_col_names))
    df = df.rename(columns=col_mapping)
    return df

def unit_test_rename_columns_with_prefix():
    data = {'N': [1, 2, 3], 'slope': [0.5, 0.6, 0.7], 'x-N_y-slope_abs_slope': [0.4, 0.5, 0.6], 'x-N_y-slope_abs_intercept': [1.0, 1.1, 1.2], 'intercept_true': [0.9, 1.0, 1.1]}
    df = pd.DataFrame(data)
    try:
        modified_df = rename_columns_with_prefix(df, prefix='x-N_y-slope_abs_', suffix='_r1')
        print(modified_df.columns)
    except ValueError as e:
        print(f'错误: {e}')
if (__name__ == '__main__'):
    unit_test_rename_columns_with_prefix()

def cal_mean_error(df_source: DataFrame, gt_name: str, pred_name: str, is_print=False):
    abs_error = abs((df_source[gt_name] - df_source[pred_name]))
    abs_error_mean = np.mean(abs_error)
    abs_relative_error = abs(((df_source[gt_name] - df_source[pred_name]) / df_source[gt_name]))
    abs_relative_error_mean = np.mean(abs_relative_error)
    if is_print:
        print(f'gt:{gt_name}, pred:{pred_name}, abs_error_mean: {abs_error_mean}, abs_relative_error: {abs_relative_error_mean}')
    return (abs_error_mean, abs_relative_error_mean)

def ax_default_setting(axes: (plt.Axes | list)):
    if (not isinstance(axes, list)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(which='both', color='black')
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1)
