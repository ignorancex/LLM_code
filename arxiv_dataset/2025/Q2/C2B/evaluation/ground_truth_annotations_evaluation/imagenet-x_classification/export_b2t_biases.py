import os

import pandas as pd


RESULTS_DIR = '../../../competitors/B2T-repro/detected_biases'
FILE_PREFIX = 'inx-val'
FILE_PREFIX_LEN = len(FILE_PREFIX)

bias_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith(FILE_PREFIX)])

df_bias = []

for bias_file in bias_files:
    decomposed_file_name = bias_file[:-4].split('_')
    model = '_'.join(decomposed_file_name[1:-1])
    target = decomposed_file_name[-1]

    df = pd.read_csv(os.path.join(RESULTS_DIR, bias_file), index_col=0, dtype={'Keyword': str, 'Score': float,
                                                                               'Acc.': float, 'Bias': str})

    df['model'] = model
    df['target class'] = target
    df_bias.append(df)

df_bias = pd.concat(df_bias).reset_index(drop=True)[['Keyword', 'Score', 'Acc.', 'target class', 'model']].rename(
    columns={'Keyword': 'bias keyword', 'Score': 'bias score', 'Acc.': 'TPR'})

df_bias.to_feather('b2t_biases.feather')
