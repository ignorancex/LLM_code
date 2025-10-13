
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from scalinglaw_utils.scaling_law_fiting.fitting_utils import ax_default_setting
from scalinglaw_utils.scaling_law_fiting.group_fit_warp import get_x_log_y_log, multi_fit_warp

def fit_multi_y_in_four_format(df: DataFrame, x_key: str, y_keys: list[str], is_plot=False):
    (fig, axes) = plt.subplots((3 * len(y_keys)), 4, figsize=(20, (len(y_keys) * 12)), dpi=300)
    for j in range(4):
        (x_log, y_log) = get_x_log_y_log(j)
        (df_res, fit_res) = multi_fit_warp(df, x_key, y_keys, x_log, y_log)
        for i in range(len(y_keys)):
            prefix = ((('x-' + x_key) + '_y-') + y_keys[i])
            sns.lineplot(data=df_res, x=x_key, y=y_keys[i], ax=axes[(3 * i)][j], marker='o', linestyle='dotted')
            sns.lineplot(data=df_res, x=x_key, y=(prefix + '_pred_raw'), ax=axes[(3 * i)][j])
            sns.lineplot(data=df_res, x=x_key, y=(prefix + '_residual'), ax=axes[((3 * i) + 1)][j])
            sns.lineplot(data=df_res, x=x_key, y=(prefix + '_relative_residual'), ax=axes[((3 * i) + 2)][j])
            this_fit_res = fit_res.query(f"name == '{prefix}'")
            formatted_residual_mean = f"{this_fit_res['residual_mean'].item():.2e}"
            formatted_relative_residual_mean = f"{this_fit_res['relative_residual_mean'].item():.2e}"
            axes[((3 * i) + 1)][j].set_title(f'residual mean:{formatted_residual_mean}', loc='center')
            axes[((3 * i) + 2)][j].set_title(f'relative mean: {formatted_relative_residual_mean}', loc='center')
            ax_default_setting([axes[(3 * i)][j], axes[((3 * i) + 1)][j], axes[((3 * i) + 2)][j]])
            if x_log:
                axes[(3 * i)][j].set_xscale('log')
            if y_log:
                axes[(3 * i)][j].set_yscale('log')
    plt.tight_layout()
    plt.show()
    plt.close()
    return
