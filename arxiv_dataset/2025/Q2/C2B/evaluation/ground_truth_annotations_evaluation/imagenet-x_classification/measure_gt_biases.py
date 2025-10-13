import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FACTORS = [
    "pose",
    "background",
    "pattern",
    "color",
    "smaller",
    "shape",
    "partial_view",
    "subcategory",
    "texture",
    "larger",
    "darker",
    "object_blocking",
    "person_blocking",
    "style",
    "brighter",
    "multiple_objects",
]

models = ['ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'ViT_B_16_SWAG']
MODELS_OUTPUT_PATH = '../../../base_models/imagenet-x_classification/torchvision_models/'
MIN_POSITIVES = 1
MIN_NEGATIVES = 1

df_list = []

for model in tqdm(models):
    arr = np.load(f'{MODELS_OUTPUT_PATH}/results_inx_{model}.npy')
    df = pd.DataFrame(data=arr, columns=['target', 'pred', 'per class TPR', 'class score', 'class rank', *FACTORS])
    df = df.drop(columns=['pred'])
    df['model'] = model
    df_list.append(df)

all_df = pd.concat(df_list).reset_index(drop=True)

df_list = []

for factor in tqdm(FACTORS):
    df = all_df[['target', 'per class TPR', 'class score', 'class rank', factor, 'model']]
    count = df.groupby(['model', 'target', factor], sort=True).count().values[:, 0]
    df = df.groupby(['model', 'target', factor], as_index=False, sort=True).mean()
    df['count'] = count
    df = df.melt(id_vars=['model', 'target', 'per class TPR', 'class score', 'class rank', 'count'],
                 value_vars=[factor], var_name='factor', value_name='factor class')
    df['target'] = df['target'].astype(int)
    df['factor class'] = df['factor class'].astype(int).astype(str)
    df = df.pivot(index=['model', 'target', 'factor'], columns=['factor class'],
                  values=['per class TPR', 'class score', 'class rank', 'count']).dropna().reset_index()
    df.columns = [' '.join(a).strip() for a in df.columns.to_flat_index()]
    df = df[df['count 0'] > MIN_NEGATIVES]
    df = df[df['count 1'] > MIN_POSITIVES]
    df['average per class TPR diff'] = df['per class TPR 1'] - df['per class TPR 0']  # NEGATIVE = MODEL IS BIASED AGAINST FACTOR / 0 = NO BIAS / POSITIVE = MODEL IS BIASED IN FAVOR OF FACTOR
    df['average class score diff'] = df['class score 1'] - df['class score 0']  # NEGATIVE = MODEL IS BIASED AGAINST FACTOR / 0 = NO BIAS / POSITIVE = MODEL IS BIASED IN FAVOR OF FACTOR
    df['average class rank diff'] = df['class rank 0'] - df['class rank 1']  # NEGATIVE = MODEL IS BIASED AGAINST FACTOR / 0 = NO BIAS / POSITIVE = MODEL IS BIASED IN FAVOR OF FACTOR
    df_list.append(df)

agg_df = pd.concat(df_list).reset_index(drop=True)

with open('../../../base_models/imagenet-x_classification/data/imagenet/imagenet_labels.json', 'r') as f:
    labels = json.load(f)

df_results = agg_df[['model', 'target', 'factor', 'average per class TPR diff']].copy()
df_results = df_results.rename(columns={'average per class TPR diff': 'TPR diff'})
df_results['target'] = df_results['target'].apply(lambda t: labels[t])
df_results.to_feather('gt_biases.feather')
