import numpy as np
import pandas as pd
from tqdm.auto import tqdm

b2t_df = pd.read_feather('../../../ground_truth_annotations_evaluation/imagenet-x_classification/b2t_biases.feather')

vqa_df = pd.read_feather('../compute_biases_from_clean_vqa_results/vqa_b2t_biases.feather')
vqa_df = vqa_df.melt(id_vars=['target class', 'keyword', 'present'])

vqa_df['model'] = vqa_df['variable'].apply(lambda v: v.split('-')[0])
vqa_df['variable'] = vqa_df['variable'].apply(lambda v: v.split('-')[1])
vqa_df = vqa_df.pivot(index=['target class', 'keyword', 'present', 'model'], columns='variable')['value'].reset_index().rename(columns={'correct': 'VQA TPR', 'TPR diff': 'VQA TPR diff'})
vqa_df = vqa_df[vqa_df['present'] == 1.0].rename(columns={'keyword': 'bias keyword'}).drop(columns=['present'])
vqa_df['VQA bias score'] = -vqa_df['VQA TPR diff']
vqa_df = vqa_df.drop(columns=['VQA TPR', 'VQA TPR diff'])
BIAS_THRESHOLD = 0.05

# Agreement B2T <-> VQA

df_b2t_vqa = pd.merge(b2t_df, vqa_df, how='left', on=['model', 'target class', 'bias keyword'])

agreement_records = []

for model in tqdm(df_b2t_vqa['model'].unique()):
    df_m = df_b2t_vqa[df_b2t_vqa['model'] == model]
    for target_class in df_m['target class'].unique():
        df_ta = df_m[df_m['target class'] == target_class]
        for bias_keyword in df_ta['bias keyword'].unique():
            df_ba = df_ta[df_ta['bias keyword'] == bias_keyword]
            b2t_bias_vector = np.nan_to_num(df_ba['bias score'].values)
            vqa_bias_vector = np.nan_to_num(df_ba['VQA bias score'].values)

            b2t_pos_bias = (b2t_bias_vector >= BIAS_THRESHOLD).astype(np.float32)
            b2t_no_bias = (abs(b2t_bias_vector) < BIAS_THRESHOLD).astype(np.float32)
            b2t_neg_bias = (b2t_bias_vector <= -BIAS_THRESHOLD).astype(np.float32)

            vqa_pos_bias = (vqa_bias_vector >= BIAS_THRESHOLD).astype(np.float32)
            vqa_no_bias = (abs(vqa_bias_vector) < BIAS_THRESHOLD).astype(np.float32)
            vqa_neg_bias = (vqa_bias_vector <= -BIAS_THRESHOLD).astype(np.float32)

            same_bias = b2t_pos_bias * vqa_pos_bias + b2t_neg_bias * vqa_neg_bias
            no_bias = b2t_no_bias * vqa_no_bias
            opposite_bias = b2t_pos_bias * vqa_neg_bias + b2t_neg_bias * vqa_pos_bias

            agreement = (same_bias + no_bias - opposite_bias).mean()

            agreement_records.append({
                'model': model, 'target class': target_class, 'bias keyword': bias_keyword, 'agreement': agreement
            })

df_agreement = pd.DataFrame.from_records(agreement_records)
print('Agreement C2B-Bing <-> VQA:')
print(df_agreement[['model', 'agreement']].groupby('model').mean())
