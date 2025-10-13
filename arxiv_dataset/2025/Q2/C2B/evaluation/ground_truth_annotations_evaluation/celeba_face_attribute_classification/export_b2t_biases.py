import os

import pandas as pd
from tqdm.auto import tqdm


RESULTS_DIR = '../../../competitors/B2T-repro/detected_biases'
FILE_PREFIX = 'celeba-val_facexformer_'
FILE_PREFIX_LEN = len(FILE_PREFIX)

bias_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith(FILE_PREFIX)])

df_bias = []

for bias_file in tqdm(bias_files):
    attribute = bias_file.split('.')[0][FILE_PREFIX_LEN:]
    df = pd.read_csv(os.path.join(RESULTS_DIR, bias_file), index_col=0, dtype={'Keyword': str, 'Score': float,
                                                                               'Acc.': float, 'Bias': str})

    df['target class'] = attribute
    if attribute.startswith('Not_'):
        df['target attribute'] = attribute[4:]
    else:
        df['target attribute'] = attribute
    df_bias.append(df)

df_bias = pd.concat(df_bias).reset_index(drop=True)[
    ['Keyword', 'Score', 'Acc.', 'target class', 'target attribute']].rename(
    columns={'Keyword': 'bias keyword', 'Score': 'bias score', 'Acc.': 'TXR'})

df_bias = df_bias[df_bias['target class'] == df_bias['target attribute']].drop(columns=['target class']).rename(columns={'TXR': 'TPR'})
df_bias.to_feather('b2t_biases.feather')
