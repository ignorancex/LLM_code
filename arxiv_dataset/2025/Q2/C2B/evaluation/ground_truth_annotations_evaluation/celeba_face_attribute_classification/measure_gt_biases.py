from collections import defaultdict
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


GT_FILE = '../../../base_models/celeba_face_attribute_classification/data/celebA/list_attr_celeba.txt'
RESULTS_DIR = '../../../base_models/celeba_face_attribute_classification/facexformer/model output - face attribute classification - celeba'
SPLIT_FILE = '../../../base_models/celeba_face_attribute_classification/data/celebA/list_eval_partition.txt'

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

gt_values = pd.read_csv(GT_FILE, sep=' ', header=0, index_col=False, dtype=dtypes).values[:, 1:].astype(np.int32)[split_idx]
logits = np.concatenate([np.load(os.path.join(RESULTS_DIR, f)) for f in sorted(os.listdir(RESULTS_DIR)) if f.startswith('logits-')], axis=0)[split_idx]
preds = logits >= 0.0
results = (preds - 0.5) * 2

records = []

total = len(gt_values)
for bias in tqdm(range(len(attributes))):
    rows_with_gt_bias_idx = np.argwhere(gt_values[:, bias] == 1).flatten()
    total_with = len(rows_with_gt_bias_idx)
    for target in range(len(attributes)):
        if target == bias:
            continue
        target_gt_with_gt_bias = gt_values[rows_with_gt_bias_idx, target]
        target_pred_with_gt_bias = results[rows_with_gt_bias_idx, target]

        actual_positive = target_gt_with_gt_bias == 1
        actual_negative = target_gt_with_gt_bias == -1
        predicted_positive = target_pred_with_gt_bias == 1
        predicted_negative = target_pred_with_gt_bias == -1

        TP = (actual_positive & predicted_positive).sum()
        FP = (actual_negative & ~predicted_negative).sum()
        FN = (actual_positive & ~predicted_positive).sum()
        TN = (actual_negative & predicted_negative).sum()

        assert TP + FP + FN + TN == total_with

        records.append({'target attribute': attributes[target], 'target class': attributes[target],
                        'bias attribute': attributes[bias], 'bias class': attributes[bias],
                        'TPR': TP / (TP + FN)})

    rows_without_gt_bias_idx = np.argwhere(gt_values[:, bias] == -1).flatten()
    total_without = len(rows_without_gt_bias_idx)
    assert total_with + total_without == total
    for target in range(40):
        if target == bias:
            continue
        target_gt_without_gt_bias = gt_values[rows_without_gt_bias_idx, target]
        target_pred_without_gt_bias = results[rows_without_gt_bias_idx, target]

        actual_positive = target_gt_without_gt_bias == 1
        actual_negative = target_gt_without_gt_bias == -1
        predicted_positive = target_pred_without_gt_bias == 1
        predicted_negative = target_pred_without_gt_bias == -1

        TP = (actual_positive & predicted_positive).sum()
        FP = (actual_negative & ~predicted_negative).sum()
        FN = (actual_positive & ~predicted_positive).sum()
        TN = (actual_negative & predicted_negative).sum()

        assert TP + FP + FN + TN == total_without

        records.append({'target attribute': attributes[target], 'target class': attributes[target],
                        'bias attribute': attributes[bias], 'bias class': f'Not_{attributes[bias]}',
                        'TPR': TP / (TP + FN)})

df_bias = pd.DataFrame.from_records(records)

target_attributes = df_bias['target attribute'].unique()
bias_attributes = df_bias['bias attribute'].unique()
target_classes = df_bias['target class'].unique()
bias_classes = df_bias['bias class'].unique()
idx_target_classes = {target_class: i for i, target_class in enumerate(df_bias['target class'].unique())}
idx_bias_classes = {bias_class: i for i, bias_class in enumerate(df_bias['bias class'].unique())}

gt_bias_tpr_diff_matrix = np.full((len(bias_classes), len(target_classes)), np.nan, dtype=np.float32)

for target_attribute in tqdm(target_attributes):
    df_ta = df_bias[df_bias['target attribute']==target_attribute]
    ta_target_classes = df_ta['target class'].unique()
    for bias_attribute in bias_attributes:
        df_ba = df_ta[df_ta['bias attribute']==bias_attribute]
        ba_bias_classes = df_ba['bias class'].unique()
        for target_class in ta_target_classes:
            df_tc = df_ba[df_ba['target class']==target_class]
            for bias_class in ba_bias_classes:
                idx_df_bc = df_tc['bias class'] == bias_class
                idx_df_bc_neg = ~idx_df_bc
                TPR_diff = df_tc[idx_df_bc]['TPR'].mean() - df_tc[idx_df_bc_neg]['TPR'].mean()
                gt_bias_tpr_diff_matrix[idx_bias_classes[bias_class], idx_target_classes[target_class]] = TPR_diff

gt_biases = []

for i, bias_class in enumerate(bias_classes[::2]):
    for j, target_class in enumerate(target_classes):
        if np.isnan(gt_bias_tpr_diff_matrix[::2][i, j]):
            continue
        gt_biases.append({
            'bias class': bias_class, 'target class': target_class, 'TPR diff': gt_bias_tpr_diff_matrix[::2][i, j]
        })

df_bias = pd.DataFrame.from_records(gt_biases)

df_bias.to_feather('gt_biases.feather')
