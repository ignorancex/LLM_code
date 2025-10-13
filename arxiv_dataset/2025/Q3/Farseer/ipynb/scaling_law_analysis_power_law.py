
import matplotlib.pyplot as plt
import seaborn as sns
from scalinglaw_utils.basic_fitting_tool import LinearFit
from scalinglaw_utils.scaling_law_fiting.data_filters import filter_data, calculate_differences, calculate_differences_group_D, filter_data_further
from scalinglaw_utils.scaling_law_fiting.group_fit_warp import group_fit
from scalinglaw_utils.scaling_law_fiting.fitting_utils import ax_default_setting
from scalinglaw_utils.scaling_law_fiting.read_data_big_exp import read_data_1009, read_data_1011, read_data_1030
COLOR_LIST = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'grey', 'yellow', 'cyan']

def plot_group_by_D_N_bin(df, suffix_name=''):
    filtered_data = filter_data(df)
    diffed_data = calculate_differences(filtered_data)
    diffed_data_filtered = filter_data_further(diffed_data)
    (fig, axes) = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
    sns.lineplot(data=diffed_data_filtered, x='N', y='L_diff', hue='D_N_bin_int', ax=axes[0][0], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=diffed_data_filtered, x='C', y='L_diff', hue='D_N_bin_int', ax=axes[0][1], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=diffed_data_filtered, x='N', y='Minus_L_diff', hue='D_N_bin_int', ax=axes[1][0], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=diffed_data_filtered, x='C', y='Minus_L_diff', hue='D_N_bin_int', ax=axes[1][1], marker='o', linestyle='--', palette=COLOR_LIST)
    for ax in axes[1]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax_default_setting(ax)
    for ax in axes[0]:
        ax_default_setting(ax)
    plt.savefig(('./figures/Loss_diff_groupby_DNbin%s.png' % suffix_name))

def plot_group_by_N_plot_npart(df, suffix_name='', do_plot=True, x_log=True, y_log=True):
    df_grouped_n = calculate_differences(filter_data(df), group_by='N')
    df_grouped_n = df_grouped_n.query('C>5e19 or D/N > 40')
    df_grouped_n_filtered = filter_data_further(df_grouped_n, group_name='N', D_N_filter=1)
    (df_grouped_n_fit, fit_res) = group_fit(df_grouped_n_filtered, group_key='N', x_key='D', y_key='Minus_L_diff')
    df_grouped_n_part = [df_grouped_n_fit.query('N<200000000'), df_grouped_n_fit.query('N>=200000000 and N<300000000'), df_grouped_n_fit.query('N>=300000000 and N<600000000'), df_grouped_n_fit.query('N>=600000000 and N<1200000000'), df_grouped_n_fit.query('N>=1200000000')]
    model1 = LinearFit(fit_res['N'].to_numpy()[1:], fit_res['slope_abs'].to_numpy()[1:]).fit(x_log, y_log, 'sklearn').summary()
    model2 = LinearFit(fit_res['N'].to_numpy()[1:], fit_res['intercept_abs'].to_numpy()[1:]).fit(x_log, y_log, 'sklearn').summary()
    model3 = LinearFit(fit_res['N'].to_numpy(), fit_res['pred_partial_loss_delta_mean'].to_numpy()).fit(x_log, y_log, 'sklearn').summary()
    if (not do_plot):
        print(fit_res[['N', 'slope', 'intercept', 'intercept_abs']])
        return df_grouped_n_fit
    prefix = 'N_x-D_y-Minus_L_diff'
    (fig, axes) = plt.subplots((len(df_grouped_n_part) + 2), 6, figsize=(40, 35), dpi=200)
    for i in range(len(df_grouped_n_part)):
        sns.lineplot(data=df_grouped_n_part[i], x='D', y='Minus_L_diff', hue='N_str', ax=axes[i][0], marker='o', linestyle='dotted', palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='D', y=(prefix + '_pred_raw'), hue='N_with_prefix', ax=axes[i][0], palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='D', y='Minus_L_diff', hue='N_str', ax=axes[i][1], marker='o', linestyle='dotted', palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='D', y=(prefix + '_pred_raw'), hue='N_with_prefix', ax=axes[i][1], palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='D', y=(prefix + '_relative_residual'), hue='N', ax=axes[i][2], marker='o', linestyle='--', palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='D', y=(prefix + '_pred_partial_loss'), hue='N', ax=axes[i][3], marker='o', palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='D/N', y=(prefix + '_pred_partial_loss_delta'), hue='N', ax=axes[i][4], marker='o', linestyle='--', palette=COLOR_LIST)
        sns.lineplot(data=df_grouped_n_part[i], x='C', y=(prefix + '_pred_partial_loss_delta'), hue='N', ax=axes[i][5], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=fit_res, x='N', y='slope', ax=axes[len(df_grouped_n_part)][0], marker='o', linestyle='--', label='real_slope')
    sns.lineplot(x=model1.x_raw, y=(- model1.pred_raw), ax=axes[len(df_grouped_n_part)][0], label='pred_slope')
    sns.lineplot(data=fit_res, x='N', y='slope_abs', ax=axes[len(df_grouped_n_part)][1], marker='o', linestyle='--', label='real_abs_slope')
    sns.lineplot(x=model1.x_raw, y=model1.pred_raw, ax=axes[len(df_grouped_n_part)][1], label='pred_abs_slope')
    sns.lineplot(x=model1.x_raw, y=model1.relative_residual, ax=axes[len(df_grouped_n_part)][2], marker='o', linestyle='--', label='slope_relative_error')
    sns.lineplot(data=fit_res, x='N', y='intercept_abs', ax=axes[(len(df_grouped_n_part) + 1)][0], marker='o', linestyle='--', label='intercept')
    sns.lineplot(x=model2.x_raw, y=model2.pred_raw, ax=axes[(len(df_grouped_n_part) + 1)][0], label='pred_intercept')
    sns.lineplot(data=fit_res, x='N', y='intercept_abs', ax=axes[(len(df_grouped_n_part) + 1)][1], marker='o', linestyle='--', label='abs_intercept')
    sns.lineplot(x=model2.x_raw, y=model2.pred_raw, ax=axes[(len(df_grouped_n_part) + 1)][1], label='pred_intercept')
    sns.lineplot(x=model2.x_raw, y=model2.relative_residual, ax=axes[(len(df_grouped_n_part) + 1)][2], marker='o', linestyle='--', label='intercept_relative_error')
    sns.lineplot(data=fit_res, x='N', y='pred_partial_loss_delta_mean', ax=axes[len(df_grouped_n_part)][3], marker='o', linestyle='--', label='partial_loss_delta_mean')
    sns.lineplot(x=model3.x_raw, y=model3.pred_raw, ax=axes[len(df_grouped_n_part)][3], label='pred_partial_loss_delta_mean')
    sns.lineplot(data=fit_res, x='N', y='pred_partial_loss_delta_mean', ax=axes[len(df_grouped_n_part)][4], marker='o', linestyle='--', label='partial_loss_delta_mean')
    sns.lineplot(x=model3.x_raw, y=model3.pred_raw, ax=axes[len(df_grouped_n_part)][4], label='pred_partial_loss_delta_mean')
    sns.lineplot(data=fit_res, x='N', y='relative_residual_mean', ax=axes[(len(df_grouped_n_part) + 1)][4], marker='o', linestyle='--')
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            ax_default_setting(axes[i][j])
            if ((j == 1) or (j == 3)):
                if x_log:
                    axes[i][j].set_xscale('log')
                if y_log:
                    axes[i][j].set_yscale('log')
    plt.tight_layout()
    plt.savefig(('./figures/Loss_diff_groupby_N_plot_2part%s.png' % suffix_name))
    return df_grouped_n_filtered

def plot_group_by_N_gerneral(df, fit_target='L', suffix_name='', diff_filter=''):
    df_grouped_n = calculate_differences(filter_data(df, min_D=200000000), group_by='N')
    df_grouped_n_filtered = filter_data_further(df_grouped_n, group_name='N', diff_filter=diff_filter, D_N_filter=1, drop_nan_key=fit_target)
    (fig, axes) = plt.subplots(6, 8, figsize=(40, 35), dpi=300)
    prefix = f'N_x-D_y-{fit_target}'
    for j in range(4):
        (x_log, y_log) = (False, False)
        if (j == 1):
            (x_log, y_log) = (False, True)
        if (j == 2):
            (x_log, y_log) = (True, False)
        if (j == 3):
            (x_log, y_log) = (True, True)
        (df_grouped_n_fit, fit_res) = group_fit(df_grouped_n_filtered, group_key='N', x_key='D', y_key=fit_target, compute_partial_loss=False, x_log=x_log, y_log=y_log)
        df_grouped_n_part = [df_grouped_n_fit.query('N<200000000'), df_grouped_n_fit.query('N>=200000000 and N<300000000'), df_grouped_n_fit.query('N>=300000000 and N<600000000'), df_grouped_n_fit.query('N>=600000000 and N<1200000000'), df_grouped_n_fit.query('N>=1200000000 and N<2000000000'), df_grouped_n_fit.query('N>=2000000000')]
        for i in range(len(df_grouped_n_part)):
            sns.lineplot(data=df_grouped_n_part[i], x='D', y=fit_target, hue='N_str', ax=axes[i][(2 * j)], marker='o', linestyle='dotted', palette=COLOR_LIST)
            sns.lineplot(data=df_grouped_n_part[i], x='D', y=(prefix + '_pred_raw'), hue='N_with_prefix', ax=axes[i][(2 * j)], palette=COLOR_LIST)
            sns.lineplot(data=df_grouped_n_part[i], x='D', y=(prefix + '_relative_residual'), hue='N', ax=axes[i][((2 * j) + 1)], marker='o', linestyle='--', palette=COLOR_LIST)
            ax_default_setting(axes[i][(2 * j)])
            ax_default_setting(axes[i][((2 * j) + 1)])
            if x_log:
                axes[i][(2 * j)].set_xscale('log')
            if y_log:
                axes[i][(2 * j)].set_yscale('log')
    plt.tight_layout()
    plt.savefig(('./figures/Loss_groupby_N_plot_by_part%s.png' % suffix_name))
    return df_grouped_n_filtered

def plot_group_by_N(df, suffix_name='', do_plot=True):
    df_grouped_n = calculate_differences(filter_data(df), group_by='N')
    df_grouped_n_filtered = filter_data_further(df_grouped_n, group_name='N')
    (df_grouped_n_fit, fit_res_1) = group_fit(df_grouped_n_filtered, group_key='N', x_key='D', y_key='Minus_L_diff')
    (df_grouped_n_fit, fit_res_2) = group_fit(df_grouped_n_fit, group_key='N', x_key='D/N', y_key='Minus_L_diff')
    if (not do_plot):
        return df_grouped_n_filtered
    (fig, axes) = plt.subplots(2, 3, figsize=(30, 16), dpi=500)
    sns.lineplot(data=df_grouped_n_filtered, x='D', y='L_diff', hue='N', ax=axes[0][0], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_n_fit, x='D', y='N_x-D_y-Minus_L_diff_relative_residual', hue='N', ax=axes[0][1], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_n_filtered, x='D', y='Minus_L_diff', hue='N', ax=axes[0][2], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_n_filtered, x='D/N', y='L_diff', hue='N', ax=axes[1][0], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_n_fit, x='D/N', y='N_x-D/N_y-Minus_L_diff_relative_residual', hue='N', ax=axes[1][1], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_n_filtered, x='D/N', y='Minus_L_diff', hue='N', ax=axes[1][2], marker='o', linestyle='--', palette=COLOR_LIST)
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            ax_default_setting(axes[i][j])
            if (j == (len(axes[i]) - 1)):
                axes[i][j].set_xscale('log')
                axes[i][j].set_yscale('log')
    plt.savefig(('./figures/Loss_diff_groupby_N%s.png' % suffix_name))
    return df_grouped_n_filtered

def plot_group_by_D(df, suffix_name=''):
    df_grouped_d = calculate_differences_group_D(filter_data(df))
    df_grouped_d_filtered = filter_data_further(df_grouped_d, group_name='D', diff_filter='N_diff')
    df_grouped_d_filtered['N/D'] = (1 / df_grouped_d_filtered['D/N'])
    (fig, axes) = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
    sns.lineplot(data=df_grouped_d_filtered, x='N', y='L_diff', hue='D', ax=axes[0][0], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_d_filtered, x='N/D', y='L_diff', hue='D', ax=axes[0][1], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_d_filtered, x='N', y='Minus_L_diff', hue='D', ax=axes[1][0], marker='o', linestyle='--', palette=COLOR_LIST)
    sns.lineplot(data=df_grouped_d_filtered, x='N/D', y='Minus_L_diff', hue='D', ax=axes[1][1], marker='o', linestyle='--', palette=COLOR_LIST)
    for ax in axes[0]:
        ax_default_setting(ax)
    for ax in axes[1]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax_default_setting(ax)
    plt.savefig(('./figures/Loss_diff_groupby_D%s.png' % suffix_name))

def plot_all(df, suffix_name=''):
    plot_group_by_D(df, suffix_name=suffix_name)
    plot_group_by_N(df, suffix_name=suffix_name)
    plot_group_by_D_N_bin(df, suffix_name=suffix_name)

def plot_dots(df_original, df_filtered):
    n_original = df_original['N']
    d_original = df_original['D']
    (fig, ax) = plt.subplots(dpi=300)
    ax.scatter(n_original, d_original, color='red', label='Not Filtered')
    filtered_out_indices = list((set(df_original.index) - set(df_filtered.index)))
    n_filtered_out = df_original.loc[(filtered_out_indices, 'N')]
    d_filtered_out = df_original.loc[(filtered_out_indices, 'D')]
    ax.scatter(n_filtered_out, d_filtered_out, color='gray', label='Filtered Out')
    ax.set_title('Scatter Plot of N vs D: Not Filtered vs Filtered Out')
    ax.set_xlabel('N')
    ax.set_ylabel('D')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

def plot_dots_nd(df_original, df_filtered, feature_name=''):
    (fig, ax) = plt.subplots(dpi=300)
    original_set = set(zip(df_original['N'], df_original['D']))
    filtered_set = set(zip(df_filtered['N'], df_filtered['D']))
    filtered_out_set = (original_set - filtered_set)
    (n_filtered_out, d_filtered_out) = (zip(*filtered_out_set) if filtered_out_set else ([], []))
    (n_filtered, d_filtered) = (zip(*filtered_set) if filtered_set else ([], []))
    ax.scatter(n_filtered, d_filtered, color='red', label='Not Filtered')
    ax.scatter(n_filtered_out, d_filtered_out, color='gray', label='Filtered Out')
    ax.set_title(f'{feature_name} len:{len(df_filtered)}', fontsize=8)
    ax.set_xlabel('N')
    ax.set_ylabel('D')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()
if (__name__ == '__main__'):
    df_val_1 = read_data_1009()
    df_val_2 = read_data_1011()
    df_val_3 = read_data_1030()
    plot_group_by_N_plot_npart(df_val_3, suffix_name='_raw_bpw_3', do_plot=False, x_log=True, y_log=True)
