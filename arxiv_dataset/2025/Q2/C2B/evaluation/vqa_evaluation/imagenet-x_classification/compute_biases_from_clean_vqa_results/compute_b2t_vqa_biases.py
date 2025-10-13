import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


INX_PATH = '../../../../base_models/imagenet-x_classification'

with open(f'{INX_PATH}/data/imagenet/imagenet_labels.json', 'r') as f:
    imagenet_classes = json.load(f)

class_id = {imagenet_class: i for i, imagenet_class in enumerate(imagenet_classes)}
metadata_df = pd.read_feather(f'{INX_PATH}/data/inx_dataset.feather')

pred_df = pd.DataFrame({
    'image': metadata_df['img_path'].apply(lambda path: path.split('/')[-1][:-5] + '.JPEG').values,
    'ResNet50_V2': np.load(f'{INX_PATH}/torchvision_models/results_inx_ResNet50_V2.npy')[:, 1],
    'ResNet101_V2': np.load(f'{INX_PATH}/torchvision_models/results_inx_ResNet101_V2.npy')[:, 1],
    'ResNet152_V2': np.load(f'{INX_PATH}/torchvision_models/results_inx_ResNet152_V2.npy')[:, 1],
    'ViT_B_16_SWAG': np.load(f'{INX_PATH}/torchvision_models/results_inx_ViT_B_16_SWAG.npy')[:, 1],
})

vqa_df = pd.read_feather('../concat_and_clean_vqa_results/b2t_vqa_answers.feather')
df_bias = pd.merge(vqa_df, pred_df, how='left', on=['image'])
df_bias['target'] = df_bias['target class'].apply(lambda t: class_id[t])

for model in ['ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'ViT_B_16_SWAG']:
    df_bias[model + '-correct'] = (df_bias[model] == df_bias['target']).astype(np.float32)

df_agg = df_bias.drop(columns=['raw answer', 'image', 'clean answer', 'target', 'ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'ViT_B_16_SWAG']).groupby(['target class', 'keyword', 'present'], as_index=False).mean()

for model in ['ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'ViT_B_16_SWAG']:
    TPR_diff = []

    for target_attribute in tqdm(sorted(df_bias['target class'].unique())):
        df_ta = df_bias[df_bias['target class'] == target_attribute]
        for bias_attribute in sorted(df_ta['keyword'].unique()):
            df_ba = df_ta[df_ta['keyword'] == bias_attribute]
            for bias_class in sorted(df_ba['present'].unique()):
                df_bc = df_ba[df_ba['present'] == bias_class]
                df_not_bc = df_ba[df_ba['present'] != bias_class]
                TPR_diff.append(df_bc[model + '-correct'].mean() - df_not_bc[model + '-correct'].mean())

    df_agg[model + '-TPR diff'] = TPR_diff

df_agg.to_feather('vqa_b2t_biases.feather')
