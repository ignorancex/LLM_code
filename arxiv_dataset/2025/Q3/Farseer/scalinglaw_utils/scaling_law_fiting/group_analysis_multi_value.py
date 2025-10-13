
import math
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from scalinglaw_utils.scaling_law_fiting.fitting_utils import ax_default_setting

def group_analysis_multi_value(df: DataFrame, y_keys: (str | list[str]), x_key: str, gt_key: str, group_key: str, ana_groups: list[str]=None, is_plot=True):
    g_keys = df[group_key].unique()
    g_keys.sort()
    y_keys = ([y_keys] if isinstance(y_keys, str) else y_keys)
    for y_key in y_keys:
        df[f'{y_key}_error'] = (df[gt_key] - df[y_key])
        df[f'{y_key}_relative_error'] = (df[f'{y_key}_error'] / df[gt_key])
        df[f'{y_key}_abs_error'] = abs(df[f'{y_key}_error'])
        df[f'{y_key}_abs_relative_error'] = abs(df[f'{y_key}_relative_error'])
    if is_plot:
        (fig, axes) = plt.subplots(len(g_keys), 4, figsize=(20, (len(g_keys) * 4)), dpi=300)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        plt.suptitle(f"Compare {'.'.join(y_keys)} at {x_key} group by {group_key}")
        df_grouped = [df.query(f'{group_key}=={g_keys[i]}') for i in range(len(g_keys))]
        for i in range(len(df_grouped)):
            names = ['raw', 'raw(log_log)', 'error', 'relative_error']
            sns.lineplot(data=df_grouped[i], x=x_key, y=gt_key, ax=axes[i][0], label=f'{gt_key}')
            sns.lineplot(data=df_grouped[i], x=x_key, y=gt_key, ax=axes[i][1], label=f'{gt_key}')
            assert (df_grouped[i][group_key].unique().item() == g_keys[i])
            for y in range(len(y_keys)):
                y_key = y_keys[y]
                v_keys = [y_key, y_key, f'{y_key}_{names[2]}', f'{y_key}_{names[3]}']
                labels = [y_key, y_key, f'error={gt_key}-{y_key}', f'error/{gt_key}']
                for j in range(len(v_keys)):
                    sns.lineplot(data=df_grouped[i], x=x_key, y=v_keys[j], ax=axes[i][j], marker='o', linestyle='dotted', label=labels[j])
                    if (y == 0):
                        axes[i][j].set_title(f'{names[j]}, {group_key} = {g_keys[i]}')
                ax_default_setting([axes[i][k] for k in range(len(v_keys))])
                axes[i][1].set_xscale('log')
                axes[i][1].set_yscale('log')
        plt.tight_layout()
        plt.show()
        plt.close()
    if ((ana_groups is None) or (len(ana_groups) == 0)):
        return
    names = ['error_mean', 'abs_error_mean', 'relative_error_mean', 'abs_relative_error_mean', 'sample_count']
    if is_plot:
        (fig, axes) = plt.subplots(len(ana_groups), 5, figsize=(20, (len(ana_groups) * 5)), dpi=300)
    for j in range(len(ana_groups)):
        for i in range(len(y_keys)):
            df_means = df.groupby(ana_groups[j]).agg({f'{y_keys[i]}_error': ['mean', (lambda x: x.abs().mean())], f'{y_keys[i]}_relative_error': ['mean', (lambda x: x.abs().mean())], 'N_str': 'count'}).reset_index()
            df_means.columns = [(f'{col[0]}_{col[1]}' if col[1] else col[0]) for col in df_means.columns]
            df_means = df_means.rename(columns={f'{y_keys[i]}_error_mean': 'error_mean', f'{y_keys[i]}_error_<lambda_0>': 'abs_error_mean', f'{y_keys[i]}_relative_error_mean': 'relative_error_mean', f'{y_keys[i]}_relative_error_<lambda_0>': 'abs_relative_error_mean', 'N_str_count': 'sample_count'})
            if is_plot:
                for k in range(len(names)):
                    sns.lineplot(data=df_means, x=ana_groups[j], y=names[k], ax=axes[j][k], label=y_keys[i])
                    if (i == 0):
                        axes[j][k].set_title(f'analysis:{ana_groups[j]}, {names[k]}')
                ax_default_setting([axes[j][k] for k in range(len(names))])
            else:
                print(tabulate(df_means, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.6f'))
    if is_plot:
        plt.tight_layout()
        plt.show()
        plt.close()

def analysis_multi_value(df: DataFrame, y_keys: (str | list[str]), x_key: str, gt_keys: (str | list[str]), ana_groups: list[str]=None):
    y_keys = ([y_keys] if isinstance(y_keys, str) else y_keys)
    gt_keys = ([gt_keys] if isinstance(gt_keys, str) else gt_keys)
    assert (len(y_keys) == len(gt_keys))
    (fig, axes) = plt.subplots(len(y_keys), 4, figsize=(20, (len(y_keys) * 4)), dpi=300)
    for i in range(len(y_keys)):
        df[f'{y_keys[i]}_gt_{gt_keys[i]}_error'] = (df[gt_keys[i]] - df[y_keys[i]])
        df[f'{y_keys[i]}_gt_{gt_keys[i]}_relative_error'] = (df[f'{y_keys[i]}_gt_{gt_keys[i]}_error'] / df[gt_keys[i]])
        abs_error_mean = df[f'{y_keys[i]}_gt_{gt_keys[i]}_error'].abs().mean().item()
        abs_r_error_mean = df[f'{y_keys[i]}_gt_{gt_keys[i]}_relative_error'].abs().mean().item()
        title = [f'''{y_keys[i]} vs 
{gt_keys[i]}''', f'''{y_keys[i]} vs 
{gt_keys[i]} (log,log)''', f'error={abs_error_mean}', f'relative_error={abs_r_error_mean}']
        name = [y_keys[i], y_keys[i], f'{y_keys[i]}_gt_{gt_keys[i]}_error', f'{y_keys[i]}_gt_{gt_keys[i]}_relative_error']
        sns.lineplot(data=df, x=x_key, y=gt_keys[i], ax=axes[i][0], label=gt_keys[i])
        sns.lineplot(data=df, x=x_key, y=gt_keys[i], ax=axes[i][1], label=gt_keys[i])
        for j in range(len(name)):
            sns.lineplot(data=df, x=x_key, y=name[j], ax=axes[i][j], label=y_keys[i])
            axes[i][j].set_title(f'{title[j]}')
            ax_default_setting(axes[i][j])
        axes[i][1].set_xscale('log')
        axes[i][1].set_yscale('log')
    plt.tight_layout()
    plt.show()
    plt.close()
