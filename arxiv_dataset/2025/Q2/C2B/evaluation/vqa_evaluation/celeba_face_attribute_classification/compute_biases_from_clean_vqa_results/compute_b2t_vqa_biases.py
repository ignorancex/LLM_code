from collections import defaultdict
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


GT_FILE = '../../../../base_models/celeba_face_attribute_classification/data/celebA/list_attr_celeba.txt'
RESULTS_DIR = '../../../../base_models/celeba_face_attribute_classification/facexformer/model output - face attribute classification - celeba'
SPLIT_FILE = '../../../../base_models/celeba_face_attribute_classification/data/celebA/list_eval_partition.txt'

attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
              'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
              'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

split_values = pd.read_csv(SPLIT_FILE, header=None, index_col=False, sep=' ', names=['file', 'split']).values[:, 1].astype(int)
split_dict = {'train': 0, 'val': 1, 'test': 2}
split = 'val'
split_idx = split_values == split_dict[split]
dtypes = defaultdict(lambda: np.int32)
dtypes['File_Name'] = 'str'

gt_df = pd.read_csv(GT_FILE, sep=' ', header=0, index_col=False, dtype=dtypes)[split_idx].melt(id_vars=['File_Name'], var_name='target attribute', value_name='gt')

logits = np.concatenate([np.load(os.path.join(RESULTS_DIR, f)) for f in sorted(os.listdir(RESULTS_DIR)) if f.startswith('logits-')], axis=0)[split_idx]
preds = logits >= 0.0
results = (preds - 0.5) * 2
pred_df = pd.DataFrame(data=results, columns=attributes).reset_index().melt(id_vars=['index'], var_name='target attribute', value_name='pred')

gt_df['pred'] = pred_df['pred']
gt_df['correct'] = (gt_df['pred'] == gt_df['gt']).astype(np.float32)
gt_df = gt_df[gt_df['gt'] == 1.0]

vqa_df = pd.read_feather('/home/quentin.guimard/Documents/results/eval_setting_2/concat_results/celeba_b2t_vqa_biases.feather')[['image', 'keyword', 'present']]

df_1 = pd.merge(gt_df.rename(columns={'File_Name': 'image'}), vqa_df, on=['image'])

df_keywords = pd.read_feather('/home/quentin.guimard/Documents/competitors/b2t/all_keywords.feather')
df_keywords = df_keywords[df_keywords['dataset'] == 'celeba'][['target', 'Keyword']].rename(columns={'target': 'target attribute', 'Keyword': 'keyword'})

df_bias = pd.merge(df_1, df_keywords, on=['target attribute', 'keyword'])

df_agg = df_bias.drop(columns=['image']).groupby(['target attribute', 'keyword', 'present'], as_index=False).mean()

TPR_diff = []

for target_attribute in tqdm(sorted(df_bias['target attribute'].unique())):
    df_ta = df_bias[df_bias['target attribute'] == target_attribute]
    for bias_attribute in sorted(df_ta['keyword'].unique()):
        df_ba = df_ta[df_ta['keyword'] == bias_attribute]
        for bias_class in sorted(df_ba['present'].unique()):
            df_bc = df_ba[df_ba['present'] == bias_class]
            df_not_bc = df_ba[df_ba['present'] != bias_class]
            TPR_diff.append(df_bc['correct'].mean() - df_not_bc['correct'].mean())

df_agg['TPR diff'] = TPR_diff
df_agg.to_feather('vqa_b2t_biases.feather')
