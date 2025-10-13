import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


base_output_folder = './model output - face attribute classification - '
bing_output_folder = base_output_folder + 'bing'
cc12m_output_folder = base_output_folder + 'cc12m'
attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
              'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
              'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
attributes_idx = {attribute: i for i, attribute in enumerate(attributes)}

# MEASURE C2B-BING BIASES

df_list = []

for target_attribute in tqdm(sorted(os.listdir(bing_output_folder))):
    for target_class in sorted(os.listdir(os.path.join(bing_output_folder, target_attribute))):
        for bias_attribute in sorted(os.listdir(os.path.join(bing_output_folder, target_attribute, target_class))):
            for bias_class in sorted(os.listdir(os.path.join(bing_output_folder, target_attribute, target_class, bias_attribute))):
                logits = np.load(os.path.join(bing_output_folder, target_attribute, target_class, bias_attribute, bias_class, 'logits-00000.npy'))[:, attributes_idx[target_attribute]]
                preds = (logits >= 0.0).astype(np.float32)
                if target_attribute == target_class or (target_attribute == 'No_Beard' and target_class == 'Shave'):
                    expected = 1.0
                else:
                    expected = 0.0
                correct = (preds == expected).astype(np.float32)
                df_list.append(pd.DataFrame(
                    {'target attribute': target_attribute, 'target class': target_class, 'bias attribute': bias_attribute,
                     'bias class': bias_class, 'expected': expected, 'pred': preds, 'correct': correct}
                ))

all_df = pd.concat(df_list)

df_tpr = all_df[all_df['expected'] == 1.0].drop(columns=['expected', 'pred', 'target class']).groupby(['target attribute', 'bias attribute', 'bias class'], as_index=False).mean()

df_bias = df_tpr.rename(columns={'correct': 'TPR'})

tpr_diff = []

for target_attribute in tqdm(df_bias['target attribute'].unique()):
    df_ta = df_bias[df_bias['target attribute'] == target_attribute]
    for bias_attribute in df_ta['bias attribute'].unique():
        df_ba = df_ta[df_ta['bias attribute'] == bias_attribute]
        for bias_class in df_ba['bias class'].unique():
            df_bc = df_ba[df_ba['bias class'] == bias_class]
            df_other_bc = df_ba[df_ba['bias class'] != bias_class]
            df_bc_tpr = df_bc['TPR'].mean()
            df_other_bc_tpr = df_other_bc['TPR'].mean()
            tpr_diff.append(df_bc_tpr - df_other_bc_tpr)

df_bias['TPR diff'] = tpr_diff
df_bias.to_feather('c2b-bing_biases.feather')

# MEASURE C2B-CC12M BIASES

df_list = []

for target_attribute in tqdm(sorted(os.listdir(cc12m_output_folder))):
    for target_class in sorted(os.listdir(os.path.join(cc12m_output_folder, target_attribute))):
        for bias_attribute in sorted(os.listdir(os.path.join(cc12m_output_folder, target_attribute, target_class))):
            for bias_class in sorted(os.listdir(os.path.join(cc12m_output_folder, target_attribute, target_class, bias_attribute))):
                logits = np.load(os.path.join(cc12m_output_folder, target_attribute, target_class, bias_attribute, bias_class, 'logits-00000.npy'))[:, attributes_idx[target_attribute]]
                preds = (logits >= 0.0).astype(np.float32)
                if target_attribute == target_class or (target_attribute == 'No_Beard' and target_class == 'Shave'):
                    expected = 1.0
                else:
                    expected = 0.0
                correct = (preds == expected).astype(np.float32)
                df_list.append(pd.DataFrame(
                    {'target attribute': target_attribute, 'target class': target_class, 'bias attribute': bias_attribute,
                     'bias class': bias_class, 'expected': expected, 'pred': preds, 'correct': correct}
                ))

all_df = pd.concat(df_list)

df_tpr = all_df[all_df['expected'] == 1.0].drop(columns=['expected', 'pred', 'target class']).groupby(['target attribute', 'bias attribute', 'bias class'], as_index=False).mean()

df_bias = df_tpr.rename(columns={'correct': 'TPR'})

tpr_diff = []

for target_attribute in tqdm(df_bias['target attribute'].unique()):
    df_ta = df_bias[df_bias['target attribute'] == target_attribute]
    for bias_attribute in df_ta['bias attribute'].unique():
        df_ba = df_ta[df_ta['bias attribute'] == bias_attribute]
        for bias_class in df_ba['bias class'].unique():
            df_bc = df_ba[df_ba['bias class'] == bias_class]
            df_other_bc = df_ba[df_ba['bias class'] != bias_class]
            df_bc_tpr = df_bc['TPR'].mean()
            df_other_bc_tpr = df_other_bc['TPR'].mean()
            tpr_diff.append(df_bc_tpr - df_other_bc_tpr)

df_bias['TPR diff'] = tpr_diff
df_bias.to_feather('c2b-cc12m_biases.feather')
