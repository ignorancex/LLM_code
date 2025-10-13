import json

import pandas as pd
from tqdm.auto import tqdm


attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
              'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
              'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

df_gt = pd.read_feather('gt_biases.feather')
GT_BIAS_THRESHOLD = 0.05
C2B_BIAS_THRESHOLD = 0.05
B2T_BIAS_THRESHOLD = 0.05

# C2B eval

with open('matched_pairs_GT-C2B.json', 'r') as f:
    matched_pairs = json.load(f)

results_records = []

## C2B-Bing eval

df_c2b = pd.read_feather('../../../C2B-pipeline/bias_measurement/face attribute classification/facexformer/c2b-bing_biases.feather')
results = {}

for target_attribute in tqdm(attributes):
    attribute_pairs = matched_pairs[target_attribute]
    pairs_gt_to_c2b = {gt: c2b for (gt, c2b) in attribute_pairs}
    pairs_c2b_to_gt = {c2b: gt for (gt, c2b) in attribute_pairs}

    df_gt_ta = df_gt[df_gt['target class'] == target_attribute]
    df_c2b_ta = df_c2b[df_c2b['target attribute'] == target_attribute]

    df_gt_ta_bias = df_gt_ta[abs(df_gt_ta['TPR diff']) > GT_BIAS_THRESHOLD]
    df_c2b_ta_bias = df_c2b_ta[abs(df_c2b_ta['TPR diff']) > C2B_BIAS_THRESHOLD]

    gt_detected = df_gt_ta_bias['bias class'].values.tolist()
    gt_scores = df_gt_ta_bias['TPR diff'].values.tolist()

    c2b_detected = [f'{bias_attribute} -- {bias_class}' for (bias_attribute, bias_class) in
                     zip(df_c2b_ta_bias['bias attribute'].values.tolist(),
                         df_c2b_ta_bias['bias class'].values.tolist())]
    c2b_scores = df_c2b_ta_bias['TPR diff'].values.tolist()

    results[target_attribute] = {
        'gt bias with no potential match': 0,
        'gt bias with potential match not detected': 0,
        'gt bias with potential match detected - same sign': 0,
        'gt bias with potential match detected - opposite sign': 0,
        'detected bias with potential match but not there': 0,
        'detected bias with no potential match': 0
    }

    for i, gt_bias in enumerate(gt_detected):
        gt_bias_score = gt_scores[i]
        if gt_bias in pairs_gt_to_c2b:
            corresponding_c2b = pairs_gt_to_c2b[gt_bias]
            if corresponding_c2b in c2b_detected:
                j = c2b_detected.index(corresponding_c2b)
                c2b_bias_score = c2b_scores[j]
                if gt_bias_score * c2b_bias_score >= 0:
                    results[target_attribute]['gt bias with potential match detected - same sign'] += 1
                else:
                    results[target_attribute]['gt bias with potential match detected - opposite sign'] += 1
            else:
                results[target_attribute]['gt bias with potential match not detected'] += 1
        else:
            results[target_attribute]['gt bias with no potential match'] += 1

    for j, c2b_bias in enumerate(c2b_detected):
        c2b_bias_score = c2b_scores[j]
        if c2b_bias in pairs_c2b_to_gt:
            corresponding_gt = pairs_c2b_to_gt[c2b_bias]
            if corresponding_gt in gt_detected:
                pass  # already handled - do not count twice
            else:
                results[target_attribute]['detected bias with potential match but not there'] += 1
        else:
            results[target_attribute]['detected bias with no potential match'] += 1

for target_attribute in tqdm(attributes):
    all_cases_dict = results[target_attribute]
    record = {'method': 'C2B-Bing', 'target attribute': target_attribute, **all_cases_dict}
    results_records.append(record)

## C2B-CC12M eval

df_c2b = pd.read_feather('../../../C2B-pipeline/bias_measurement/face attribute classification/facexformer/c2b-cc12m_biases.feather')
results = {}

for target_attribute in tqdm(attributes):
    attribute_pairs = matched_pairs[target_attribute]
    pairs_gt_to_c2b = {gt: c2b for (gt, c2b) in attribute_pairs}
    pairs_c2b_to_gt = {c2b: gt for (gt, c2b) in attribute_pairs}

    df_gt_ta = df_gt[df_gt['target class'] == target_attribute]
    df_c2b_ta = df_c2b[df_c2b['target attribute'] == target_attribute]

    df_gt_ta_bias = df_gt_ta[abs(df_gt_ta['TPR diff']) > GT_BIAS_THRESHOLD]
    df_c2b_ta_bias = df_c2b_ta[abs(df_c2b_ta['TPR diff']) > C2B_BIAS_THRESHOLD]

    gt_detected = df_gt_ta_bias['bias class'].values.tolist()
    gt_scores = df_gt_ta_bias['TPR diff'].values.tolist()

    c2b_detected = [f'{bias_attribute} -- {bias_class}' for (bias_attribute, bias_class) in
                     zip(df_c2b_ta_bias['bias attribute'].values.tolist(),
                         df_c2b_ta_bias['bias class'].values.tolist())]
    c2b_scores = df_c2b_ta_bias['TPR diff'].values.tolist()

    results[target_attribute] = {
        'gt bias with no potential match': 0,
        'gt bias with potential match not detected': 0,
        'gt bias with potential match detected - same sign': 0,
        'gt bias with potential match detected - opposite sign': 0,
        'detected bias with potential match but not there': 0,
        'detected bias with no potential match': 0
    }

    for i, gt_bias in enumerate(gt_detected):
        gt_bias_score = gt_scores[i]
        if gt_bias in pairs_gt_to_c2b:
            corresponding_c2b = pairs_gt_to_c2b[gt_bias]
            if corresponding_c2b in c2b_detected:
                j = c2b_detected.index(corresponding_c2b)
                c2b_bias_score = c2b_scores[j]
                if gt_bias_score * c2b_bias_score >= 0:
                    results[target_attribute]['gt bias with potential match detected - same sign'] += 1
                else:
                    results[target_attribute]['gt bias with potential match detected - opposite sign'] += 1
            else:
                results[target_attribute]['gt bias with potential match not detected'] += 1
        else:
            results[target_attribute]['gt bias with no potential match'] += 1

    for j, c2b_bias in enumerate(c2b_detected):
        c2b_bias_score = c2b_scores[j]
        if c2b_bias in pairs_c2b_to_gt:
            corresponding_gt = pairs_c2b_to_gt[c2b_bias]
            if corresponding_gt in gt_detected:
                pass  # already handled - do not count twice
            else:
                results[target_attribute]['detected bias with potential match but not there'] += 1
        else:
            results[target_attribute]['detected bias with no potential match'] += 1

for target_attribute in tqdm(attributes):
    all_cases_dict = results[target_attribute]
    record = {'method': 'C2B-CC12M', 'target attribute': target_attribute, **all_cases_dict}
    results_records.append(record)

# B2T eval

with open('matched_pairs_GT-B2T.json', 'r') as f:
    matched_pairs = json.load(f)

df_b2t = pd.read_feather('b2t_biases.feather')
results = {}

for target_attribute in tqdm(attributes):
    attribute_pairs = matched_pairs[target_attribute]
    pairs_gt_to_b2t = {gt: b2t for (gt, b2t) in attribute_pairs}
    pairs_b2t_to_gt = {b2t: gt for (gt, b2t) in attribute_pairs}

    df_gt_ta = df_gt[df_gt['target class'] == target_attribute]
    df_b2t_ta = df_b2t[df_b2t['target attribute'] == target_attribute]

    df_gt_ta_bias = df_gt_ta[abs(df_gt_ta['TPR diff']) > GT_BIAS_THRESHOLD]
    df_b2t_ta_bias = df_b2t_ta[abs(df_b2t_ta['bias score']) > B2T_BIAS_THRESHOLD]

    gt_detected = df_gt_ta_bias['bias class'].values.tolist()
    gt_scores = df_gt_ta_bias['TPR diff'].values.tolist()

    b2t_detected = df_b2t_ta_bias['bias keyword'].values.tolist()
    b2t_scores = df_b2t_ta_bias['bias score'].values.tolist()

    results[target_attribute] = {
        'gt bias with no potential match': 0,
        'gt bias with potential match not detected': 0,
        'gt bias with potential match detected - same sign': 0,
        'gt bias with potential match detected - opposite sign': 0,
        'detected bias with potential match but not there': 0,
        'detected bias with no potential match': 0
    }

    for i, gt_bias in enumerate(gt_detected):
        gt_bias_score = gt_scores[i]
        if gt_bias in pairs_gt_to_b2t:
            corresponding_b2t = pairs_gt_to_b2t[gt_bias]
            if corresponding_b2t in b2t_detected:
                j = b2t_detected.index(corresponding_b2t)
                b2t_bias_score = b2t_scores[j]
                if gt_bias_score * b2t_bias_score <= 0:  # ACTUALLY BIAS SCORE IS POSITIVE IN B2T!!!!!!
                    results[target_attribute]['gt bias with potential match detected - same sign'] += 1
                else:
                    results[target_attribute]['gt bias with potential match detected - opposite sign'] += 1
            else:
                results[target_attribute]['gt bias with potential match not detected'] += 1
        else:
            results[target_attribute]['gt bias with no potential match'] += 1

    for j, b2t_bias in enumerate(b2t_detected):
        b2t_bias_score = b2t_scores[j]
        if b2t_bias in pairs_b2t_to_gt:
            corresponding_gt = pairs_b2t_to_gt[b2t_bias]
            if corresponding_gt in gt_detected:
                pass  # already handled - do not count twice
            else:
                results[target_attribute]['detected bias with potential match but not there'] += 1
        else:
            results[target_attribute]['detected bias with no potential match'] += 1

for target_attribute in tqdm(attributes):
    all_cases_dict = results[target_attribute]
    record = {'method': 'B2T', 'target attribute': target_attribute, **all_cases_dict}
    results_records.append(record)

# Final metrics

df_results = pd.DataFrame.from_records(results_records)

df_results_gt = df_results.drop(columns=['target attribute', 'detected bias with potential match but not there', 'detected bias with no potential match'])
df_results_gt['miss'] = df_results_gt['gt bias with no potential match'] + df_results_gt['gt bias with potential match not detected']
df_results_gt = df_results_gt.drop(columns=['gt bias with no potential match', 'gt bias with potential match not detected']).rename(columns={'gt bias with potential match detected - same sign': 'hit', 'gt bias with potential match detected - opposite sign': 'false hit'})
gt_columns = df_results_gt.columns[1:]
gt_biases = df_results_gt[gt_columns].values.sum(axis=1)
for gt_col in gt_columns:
    df_results_gt[gt_col] = 100 * df_results_gt[gt_col] / gt_biases
print('GT -> Detected')
print(df_results_gt.groupby('method').mean())

df_results_detected = df_results.drop(columns=['target attribute', 'gt bias with no potential match', 'gt bias with potential match not detected'])
df_results_detected['miss'] = df_results_detected['detected bias with potential match but not there'] + df_results_detected['detected bias with no potential match']
df_results_detected = df_results_detected.drop(columns=['detected bias with potential match but not there', 'detected bias with no potential match']).rename(columns={'gt bias with potential match detected - same sign': 'hit', 'gt bias with potential match detected - opposite sign': 'false hit'})
detected_columns = df_results_detected.columns[1:]
detected_biases = df_results_detected[detected_columns].values.sum(axis=1)
for detected_col in detected_columns:
    df_results_detected[detected_col] = 100 * df_results_detected[detected_col] / detected_biases
print('Detected -> GT')
print(df_results_detected.groupby('method').mean())
