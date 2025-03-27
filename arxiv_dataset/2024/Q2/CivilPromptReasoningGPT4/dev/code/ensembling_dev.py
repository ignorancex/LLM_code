# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:00:34 2024

@author: dansc
"""


from sklearn.metrics import f1_score
import sys

sys.path.append('../functions')
#homebrew
from load_data_dict import load_data_dict
from calculations import (
    scores,
    extract_output_values_from_json_file,
    re_scan,
    list_to_csv,
    ensemble_w_cutoff
)

data_dict = load_data_dict('../../data/dev.csv')

ground_truth = []
for tf in data_dict['label']:
    if tf == 'TRUE':
        ground_truth.append(int(1))
    else:
        ground_truth.append(int(0))


tf_pattern = '(TRUE|FALSE)'

oneShot = extract_output_values_from_json_file('../data/one-shot.out')
oneShot = re_scan(tf_pattern, oneShot)

oneShot_COT = extract_output_values_from_json_file('../data/one-shot_COT_TF.out')
oneShot_COT = re_scan(tf_pattern, oneShot_COT)

fewShot = extract_output_values_from_json_file('../data/few-shot.out')
fewShot = re_scan(tf_pattern, fewShot)

fewShot_COT = extract_output_values_from_json_file('../data/few-shot_COT_TF.out')
fewShot_COT = re_scan(tf_pattern, fewShot_COT)

RAG_no_COT = extract_output_values_from_json_file('../data/few-shot_RAG.out')
RAG_no_COT = re_scan(tf_pattern, RAG_no_COT)

RAG_COT= extract_output_values_from_json_file('../data/few-shot_RAG_COT_TF.out')
RAG_COT = re_scan(tf_pattern, RAG_COT)

# i manually put them in the order of highest  F1
results_list = [RAG_COT, oneShot_COT, fewShot, RAG_no_COT, fewShot_COT, oneShot]


cutoff_list = [.3,.4,.5,.6,.7,.8,.9,1]

best_f1 = 0
best_model = None
best_cutoff = None


from itertools import combinations

# Dictionary mapping names to variables
model_dict = {
    'RAG_COT': RAG_COT,
    'oneShot_COT': oneShot_COT,
    'fewShot': fewShot,
    'RAG_no_COT': RAG_no_COT,
    'fewShot_COT': fewShot_COT,
    'oneShot': oneShot
}

# combination loop
for cutoff in cutoff_list:
    for r in range(1, len(model_dict) + 1):
        for model_combo in combinations(model_dict, r):
            # Use the names from the combination to get the actual lists
            model_lists = [model_dict[name] for name in model_combo]
            ensemble_preds = ensemble_w_cutoff(model_lists, cutoff)
            f1 = f1_score(ground_truth, ensemble_preds, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_combination = model_combo  # This now holds the names
                best_cutoff = cutoff

# After the loop
print(f'Best F1 Score: {best_f1:.4f}')
print(f'Best Combination: {best_combination}')
print(f'Best Cutoff: {best_cutoff}')

best_list = [RAG_COT, oneShot_COT,  fewShot_COT, oneShot]
best_cutoff
best_list = []
for key in best_combination:
    best_list.append(model_dict[key])

best_ens_results = ensemble_w_cutoff(best_list, best_cutoff)
list_to_csv(path = '../data/dev_final1.csv', lst = best_ens_results)
scores(ground_truth, best_ens_results)

