import os

import pandas as pd
from tqdm.auto import tqdm


gt_keywords_folder = './detected_biases'
split = 'val'

df_list = []

for file in tqdm(os.listdir(gt_keywords_folder)):
    if file.split('-')[1].startswith(split):
        df = pd.read_csv(os.path.join(gt_keywords_folder, file))[['Keyword', 'Score', 'Acc.']].copy()
        dataset = file.split('-')[0]
        decomposed_file_name = file[:-4].split('_')
        if dataset == 'celeba':
            model = decomposed_file_name[1]
            target = '_'.join(decomposed_file_name[2:])
        elif dataset == 'inx':
            model = '_'.join(decomposed_file_name[1:-1])
            target = decomposed_file_name[-1]
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        df['dataset'] = dataset
        df['model'] = model
        df['target'] = target
        df_list.append(df)

df = pd.concat(df_list).reset_index(drop=True)
df.to_feather('all_keywords.feather')
