
import pandas as pd
import numpy as np
from tqdm import tqdm

def fix_counts(df, side, group_var, group):
    res = {'agree', 'disagree', 'neutral'}.difference(set(df['label']))
    if bool(res):
        for item in res:
            df = pd.concat([df, pd.DataFrame({group_var:group, 'pol_label_gpt':side, 'label': [item], 'proportion':[0.0]})])
        return df
    else:
        return df

def compute_bias(df, group_var, group):
    bias_df = pd.DataFrame(df.groupby([group_var, 'pol_label_gpt'])['label'].value_counts(normalize=True)).rename(columns={'label' : 'proportion'}).reset_index()

    right_df = bias_df[(bias_df[group_var]==group) & (bias_df['pol_label_gpt'] == 'right' )]
    left_df = bias_df[(bias_df[group_var]==group) & (bias_df['pol_label_gpt'] == 'left' )]

    right_df = fix_counts(right_df, 'right', group_var, group)
    left_df = fix_counts(left_df, 'left', group_var, group)

    try: 
        right_bias = right_df.loc[right_df['label']=='agree', 'proportion'].item() - right_df.loc[right_df['label']=='disagree', 'proportion'].item() 
        left_bias = left_df.loc[left_df['label']=='agree', 'proportion'].item() - left_df.loc[left_df['label']=='disagree', 'proportion'].item() 
        sum_bias = (right_bias-left_bias)/2.0
    except ValueError:
       sum_bias=np.nan

    return sum_bias

def confidence_intervals(stats):
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return lower, upper


def get_bootstrapped_ci(data_1, data_2, n_iterations):
    stats = []
    org_stat = compute_bias(data_1, None, None) - compute_bias(data_2, None, None)
    for i in tqdm(range(n_iterations)):
        sample_1 = data_1.sample(frac=1.0, replace=True)
        sample_2 = data_2.sample(frac=1.0, replace=True)
        bias = compute_bias(sample_1, None, None) - compute_bias(sample_2, None, None)
        if not pd.isna(bias):
            stats.append(bias)  

    lower, upper = confidence_intervals(stats)

    return org_stat, lower, upper