import numpy as np
import pandas as pd


c2b_bing_df = pd.read_feather('../../../../C2B-pipeline/bias_measurement/face attribute classification/facexformer/c2b-bing_biases.feather')
c2b_cc12m_df = pd.read_feather('../../../../C2B-pipeline/bias_measurement/face attribute classification/facexformer/c2b-cc12m_biases.feather')

vqa_df = pd.read_feather('../compute_biases_from_clean_vqa_results/vqa_c2b_biases.feather')
vqa_df = vqa_df.rename(columns={'bias class answer': 'bias class', 'correct': 'VQA TPR', 'TPR diff': 'VQA TPR diff'})

BIAS_THRESHOLD = 0.05

# Agreement C2B-Bing <-> VQA

df_bing_vqa = pd.merge(c2b_bing_df, vqa_df, how='left', on=['target attribute', 'bias attribute', 'bias class'])
agreement_records = []

for target_attribute in df_bing_vqa['target attribute'].unique():
    df_ta = df_bing_vqa[df_bing_vqa['target attribute'] == target_attribute]
    for bias_attribute in df_ta['bias attribute'].unique():
        df_ba = df_ta[df_ta['bias attribute'] == bias_attribute]
        bing_bias_vector = np.nan_to_num(df_ba['TPR diff'].values)
        vqa_bias_vector = np.nan_to_num(df_ba['VQA TPR diff'].values)

        bing_pos_bias = (bing_bias_vector >= BIAS_THRESHOLD).astype(np.float32)
        bing_no_bias = (abs(bing_bias_vector) < BIAS_THRESHOLD).astype(np.float32)
        bing_neg_bias = (bing_bias_vector <= -BIAS_THRESHOLD).astype(np.float32)

        vqa_pos_bias = (vqa_bias_vector >= BIAS_THRESHOLD).astype(np.float32)
        vqa_no_bias = (abs(vqa_bias_vector) < BIAS_THRESHOLD).astype(np.float32)
        vqa_neg_bias = (vqa_bias_vector <= -BIAS_THRESHOLD).astype(np.float32)

        same_bias = bing_pos_bias * vqa_pos_bias + bing_neg_bias * vqa_neg_bias
        no_bias = bing_no_bias * vqa_no_bias
        opposite_bias = bing_pos_bias * vqa_neg_bias + bing_neg_bias * vqa_pos_bias

        agreement = (same_bias + no_bias - opposite_bias).mean()

        agreement_records.append({
            'target attribute': target_attribute, 'bias attribute': bias_attribute, 'agreement': agreement
        })

df_agreement = pd.DataFrame.from_records(agreement_records)
print('Agreement C2B-Bing <-> VQA:')
print(df_agreement['agreement'].mean())

# Agreement C2B-CC12M <-> VQA

df_cc12m_vqa = pd.merge(c2b_cc12m_df, vqa_df, how='left', on=['target attribute', 'bias attribute', 'bias class'])
agreement_records = []

for target_attribute in df_cc12m_vqa['target attribute'].unique():
    df_ta = df_cc12m_vqa[df_cc12m_vqa['target attribute'] == target_attribute]
    for bias_attribute in df_ta['bias attribute'].unique():
        df_ba = df_ta[df_ta['bias attribute'] == bias_attribute]
        cc12m_bias_vector = np.nan_to_num(df_ba['TPR diff'].values)
        vqa_bias_vector = np.nan_to_num(df_ba['VQA TPR diff'].values)

        cc12m_pos_bias = (cc12m_bias_vector >= BIAS_THRESHOLD).astype(np.float32)
        cc12m_no_bias = (abs(cc12m_bias_vector) < BIAS_THRESHOLD).astype(np.float32)
        cc12m_neg_bias = (cc12m_bias_vector <= -BIAS_THRESHOLD).astype(np.float32)

        vqa_pos_bias = (vqa_bias_vector >= BIAS_THRESHOLD).astype(np.float32)
        vqa_no_bias = (abs(vqa_bias_vector) < BIAS_THRESHOLD).astype(np.float32)
        vqa_neg_bias = (vqa_bias_vector <= -BIAS_THRESHOLD).astype(np.float32)

        same_bias = cc12m_pos_bias * vqa_pos_bias + cc12m_neg_bias * vqa_neg_bias
        no_bias = cc12m_no_bias * vqa_no_bias
        opposite_bias = cc12m_pos_bias * vqa_neg_bias + cc12m_neg_bias * vqa_pos_bias

        agreement = (same_bias + no_bias - opposite_bias).mean()

        agreement_records.append({
            'target attribute': target_attribute, 'bias attribute': bias_attribute, 'agreement': agreement
        })

df_agreement = pd.DataFrame.from_records(agreement_records)
print('Agreement C2B-CC12M <-> VQA:')
print(df_agreement['agreement'].mean())
