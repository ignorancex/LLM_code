
import numpy as np

def filter_data(df, min_D=15000000000, min_N=200000000, D_N_filter=0):
    "\n    Filters the original DataFrame to retain only relevant columns and prepares it for analysis.\n\n    Parameters:\n    - df (DataFrame): Original DataFrame containing various columns.\n`\n    Returns:\n    - filtered_df (DataFrame): Filtered DataFrame with columns ['D', 'N', 'M', 'D/N', 'round(D/N)', 'ND', 'C', 'L']\n    "
    bins = [0, 1, 1.3, 2, 3, 4, 5.5, 7.5, 10.5, 15, 21, 30, 41, 60, 81, 114, 162, 228, 330, 460, 650, 1000, 2000]
    labels = [1, 1.25, 1.75, 2.5, 3.5, 5, 7, 10, 14, 20, 28, 40, 57, 80, 113, 160, 225, 320, 450, 640, 910, 1285]
    filtered_df = df[['D', 'N', 'M', 'D/N', 'round(D/N)', 'ND', 'C', 'L']]
    filtered_df = filtered_df.query(f'D>{min_D} or N>{min_N}').copy()
    filtered_df.loc[(:, 'round(D/N)')] = filtered_df['round(D/N)'].round(1)
    filtered_df.loc[(:, 'N_with_prefix')] = filtered_df['N'].apply((lambda x: ('pred_' + str(int(x)))))
    filtered_df.loc[(:, 'N_str')] = filtered_df['N'].apply((lambda x: ('real_' + str(int(x)))))
    filtered_df.loc[(:, 'D_round')] = filtered_df['D'].apply((lambda x: float('{:.3e}'.format(x))))
    filtered_df['D_log10'] = np.log10(filtered_df['D'])
    filtered_df = D_N_bin_cluster(filtered_df)
    filtered_df['D_N_bin_int'] = filtered_df['D_N_barrier'].round(1)
    if (D_N_filter > 0):
        filtered_df = filtered_df.query(f'D_N_barrier >= {D_N_filter}')
    filtered_df = filtered_df.drop_duplicates()
    return filtered_df

def calculate_differences(df, group_by: str='D_N_bin_int', sort_by: str='D', v_keys: (str | list[str])='L', suffix: str=''):
    df_with_differences = df.copy()
    grouped_sorted = df_with_differences.sort_values([group_by, sort_by]).groupby(group_by)
    df_with_differences[f'{sort_by}_diff{suffix}'] = grouped_sorted[sort_by].pct_change()
    df_with_differences[f'{sort_by}_origin{suffix}'] = grouped_sorted[sort_by].shift(1)
    v_keys = (v_keys if isinstance(v_keys, list) else [v_keys])
    for v_key in v_keys:
        df_with_differences[f'{v_key}_diff'] = grouped_sorted[v_key].diff()
        df_with_differences[f'{v_key}_prev'] = grouped_sorted[v_key].shift(1)
        df_with_differences[f'Minus_{v_key}_diff'] = (- df_with_differences[f'{v_key}_diff'])
    return df_with_differences

def calculate_differences_group_D(df):
    df_with_differences = df.copy()
    grouped_sorted = df_with_differences.sort_values(['D', 'N']).groupby('D')
    df_with_differences['N_diff'] = grouped_sorted['N'].pct_change()
    df_with_differences['L_diff'] = grouped_sorted['L'].diff()
    df_with_differences['Minus_L_diff'] = (- df_with_differences['L_diff'])
    return df_with_differences

def filter_data_further(df, group_name='D_N_bin', diff_filter='D_diff', D_N_filter=0, drop_nan_key='L_diff', min_group_num=6, N_filters: list=None, is_print=False):
    '\n    对数据进行进一步过滤,主要包含以下几个步骤:\n    1. 去除指定列(drop_nan_key)中的NaN值\n    2. 根据diff_filter参数过滤数据:\n       - 如果是"D_diff": 保留D_diff在0.35-0.45之间的数据\n       - 如果是"N_diff": 保留N_diff在0.35-0.45之间的数据\n    3. 根据D_N_filter参数过滤D/N比值\n    4. 应用额外的N值过滤条件(如果提供)\n    5. 去除每个分组中数据点少于min_group_num的组\n\n    参数:\n    - df (DataFrame): 输入的数据框\n    - group_name (str): 分组的列名,默认为\'D_N_bin\'\n    - diff_filter (str): 差分过滤类型,可选"D_diff"或"N_diff"\n    - D_N_filter (float): D/N比值的最小阈值,默认为0\n    - drop_nan_key (str): 需要去除NaN值的列名,默认为"L_diff"\n    - min_group_num (int): 每个分组最少需要的数据点数,默认为6\n    - N_filters (list): 额外的N值过滤条件列表,默认为None\n    - is_print (bool): 是否打印过滤信息,默认为False\n\n    返回:\n    - DataFrame: 过滤后的数据框\n    '
    df_no_nan = df.copy()
    if (drop_nan_key != ''):
        filtered_out_nan = df_no_nan[df_no_nan[drop_nan_key].isna()]
        if is_print:
            print(filtered_out_nan[['N', 'D', 'D_round', drop_nan_key]].sort_values(['N', 'D']))
        df_no_nan = df_no_nan.dropna(subset=[drop_nan_key])
    if (diff_filter == 'D_diff'):
        filtered_out_diff = df_no_nan[((df_no_nan['D_diff'] >= 0.45) | (df_no_nan['D_diff'] <= 0.35))]
        filtered_out_diff = filtered_out_diff[['N', 'D', 'D_round', 'D_diff']].sort_values(['N', 'D', 'D_round'])
        if is_print:
            print(f'D_diff filter out {filtered_out_diff}')
        df_no_nan = df_no_nan.query('D_diff < 0.45 and D_diff > 0.35')
    elif (diff_filter == 'N_diff'):
        filtered_out_diff = df_no_nan[((df_no_nan['N_diff'] <= 0.35) | (df_no_nan['N_diff'] >= 0.45))]
        filtered_out_diff = filtered_out_diff[['N', 'D', 'D_round', 'N_diff']].sort_values(['N', 'D', 'D_round'])
        if is_print:
            print(f'N_diff filter out {filtered_out_diff}')
        df_no_nan = df_no_nan.query('N_diff > 0.35 and N_diff < 0.45')
    if (D_N_filter > 0):
        df_no_nan = df_no_nan.query(('D_N_bin_int > %.3f' % D_N_filter))
    if (N_filters is not None):
        for N_filter in N_filters:
            df_no_nan = df_no_nan.query(N_filter)
    filtered_df = df_no_nan.groupby(group_name).filter((lambda x: (len(x) >= min_group_num)))
    return filtered_df

def D_N_bin_cluster(df_r1):
    df_r1 = df_r1.sort_values('D/N').reset_index(drop=True)
    cluster_num = 0
    cluster_labels = []
    cluster_min = None
    for (idx, row) in df_r1.iterrows():
        current_value = row['D/N']
        if (cluster_min is None):
            cluster_num += 1
            cluster_min = current_value
            cluster_labels.append(cluster_num)
        else:
            rel_diff = ((current_value - cluster_min) / cluster_min)
            if (rel_diff <= 0.1):
                cluster_labels.append(cluster_num)
            else:
                cluster_num += 1
                cluster_min = current_value
                cluster_labels.append(cluster_num)
    df_r1['cluster_num'] = cluster_labels
    df_r1['D_N_barrier'] = df_r1.groupby('cluster_num')['D/N'].transform('mean')
    return df_r1
