# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:00:34 2024

@author: dansc
"""


import sys
sys.path.append('../../functions')
#homebrew
from load_data_dict import load_test_dict
from calculations import ( 
    count_true_false,
    extract_output_values_from_json_file,
    re_scan,
    list_to_csv,
    ensemble_w_cutoff,
)

data_dict = load_test_dict('../../data/test.csv')

tf_pattern = '(TRUE|FALSE)'

oneShot = extract_output_values_from_json_file('../data/test_one-shot.out')
oneShot = re_scan(tf_pattern, oneShot)

oneShot_COT = extract_output_values_from_json_file('../data/test_one-shot_COT_TF.out')
oneShot_COT = re_scan(tf_pattern, oneShot_COT)

fewShot_COT = extract_output_values_from_json_file('../data/test_few-shot_COT_TF.out')
fewShot_COT = re_scan(tf_pattern, fewShot_COT)

RAG_COT= extract_output_values_from_json_file('../data/test_RAG_COT.out')
RAG_COT = re_scan(tf_pattern, RAG_COT)

best_list = [RAG_COT, oneShot_COT, fewShot_COT, oneShot]
best_cutoff = .5

FINAL_PREDICTIONS = ensemble_w_cutoff(best_list, best_cutoff)

count_true_false(FINAL_PREDICTIONS)

list_to_csv('./json_results/TEST/FINAL_TEST_PREDS_2024-01-29.csv', FINAL_PREDICTIONS)
