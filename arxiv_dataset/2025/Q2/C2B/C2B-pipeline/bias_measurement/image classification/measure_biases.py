import json
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


base_output_folder = '.'
models = ['ViT_B_16_SWAG', 'ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2']
file_prefix = 'results_'
file_suffix = '_retrieved_'
file_extension = '.npy'

# MEASURE C2B-BING BIASES

source = 'bing'

model_outputs = {}

for model in tqdm(models):
    model_outputs[model] = np.load(os.path.join(base_output_folder, file_prefix + model + file_suffix + source + file_extension))

with open(os.path.join(base_output_folder, f'{source}_records.json'), 'r') as f:
    meta_df = pd.DataFrame.from_records(json.load(f))

df_list = []
for model in tqdm(models):
    df = meta_df.drop(columns=['img_path'])
    df['model'] = model
    df['target_from_npy'] = model_outputs[model][:, 0]
    df['correct_from_npy'] = model_outputs[model][:, 1]
    df['class_score_from_npy'] = model_outputs[model][:, 2]
    df['class_rank_from_npy'] = model_outputs[model][:, 3]
    df_list.append(df)
bing_df = pd.concat(df_list).rename(columns={'bias_class': 'bias class'}).reset_index(drop=True)

assert np.all(bing_df['target'].values == bing_df['target_from_npy'].values)

df_tpr = bing_df.drop(columns=['target', 'target_from_npy', 'class_score_from_npy', 'class_rank_from_npy']).groupby(['model', 'target class', 'bias attribute', 'bias class'], as_index=False).mean()

df_bias = df_tpr.rename(columns={'correct_from_npy': 'TPR'})

tpr_diff = []

for model in tqdm(df_bias['model'].unique()):
    df_m = df_bias[df_bias['model'] == model]
    for target_class in df_m['target class'].unique():
        df_tc = df_m[df_m['target class'] == target_class]
        for bias_attribute in df_tc['bias attribute'].unique():
            df_ba = df_tc[df_tc['bias attribute'] == bias_attribute]
            for bias_class in df_ba['bias class'].unique():
                df_bc = df_ba[df_ba['bias class'] == bias_class]
                df_other_bc = df_ba[df_ba['bias class'] != bias_class]
                df_bc_tpr = df_bc['TPR'].mean()
                df_other_bc_tpr = df_other_bc['TPR'].mean()
                tpr_diff.append(df_bc_tpr - df_other_bc_tpr)

df_bias['TPR diff'] = tpr_diff
df_bias.to_feather('c2b-bing_biases.feather')

# MEASURE C2B-CC12M BIASES

source = 'cc12m'

model_outputs = {}

for model in tqdm(models):
    model_outputs[model] = np.load(os.path.join(base_output_folder, file_prefix + model + file_suffix + source + file_extension))

with open(os.path.join(base_output_folder, f'{source}_records.json'), 'r') as f:
    meta_df = pd.DataFrame.from_records(json.load(f))

df_list = []
for model in tqdm(models):
    df = meta_df.drop(columns=['img_path'])
    df['model'] = model
    df['target_from_npy'] = model_outputs[model][:, 0]
    df['correct_from_npy'] = model_outputs[model][:, 1]
    df['class_score_from_npy'] = model_outputs[model][:, 2]
    df['class_rank_from_npy'] = model_outputs[model][:, 3]
    df_list.append(df)
cc12m_df = pd.concat(df_list).rename(columns={'bias_class': 'bias class'}).reset_index(drop=True)

assert np.all(cc12m_df['target'].values == cc12m_df['target_from_npy'].values)

df_tpr = cc12m_df.drop(columns=['target', 'target_from_npy', 'class_score_from_npy', 'class_rank_from_npy']).groupby(['model', 'target class', 'bias attribute', 'bias class'], as_index=False).mean()

df_bias = df_tpr.rename(columns={'correct_from_npy': 'TPR'})

tpr_diff = []

for model in tqdm(df_bias['model'].unique()):
    df_m = df_bias[df_bias['model'] == model]
    for target_class in df_m['target class'].unique():
        df_tc = df_m[df_m['target class'] == target_class]
        for bias_attribute in df_tc['bias attribute'].unique():
            df_ba = df_tc[df_tc['bias attribute'] == bias_attribute]
            for bias_class in df_ba['bias class'].unique():
                df_bc = df_ba[df_ba['bias class'] == bias_class]
                df_other_bc = df_ba[df_ba['bias class'] != bias_class]
                df_bc_tpr = df_bc['TPR'].mean()
                df_other_bc_tpr = df_other_bc['TPR'].mean()
                tpr_diff.append(df_bc_tpr - df_other_bc_tpr)

df_bias['TPR diff'] = tpr_diff

df_bias.to_feather('c2b-cc12m_biases.feather')
