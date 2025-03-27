# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:13:43 2024
@author: Dan Schumacher
"""
# =============================================================================
# IMPORTS
# =============================================================================
# STANDARD IMPORTS
import sys
sys.path.append('./functions')
from load_data_dict import load_data_dict


from calculations import (
    scores,
    extract_output_values_from_json_file,
    re_scan
    )


data_dict = load_data_dict('./data/dev.csv')
ground_truth = data_dict['label']
turn_to_binary = []
for tf in ground_truth:
    if tf == 'TRUE':
        turn_to_binary.append(int(1))
    else:
        turn_to_binary.append(int(0))
ground_truth = turn_to_binary

# for RegEx
tf_pattern = '(TRUE|FALSE)'

# =============================================================================
# # ONE SHOT 
# =============================================================================
print('\nONE-SHOT ')
oneShot = extract_output_values_from_json_file('./dev/data/one-shot.out')
oneShot = re_scan(tf_pattern, oneShot)
scores(ground_truth, oneShot)

# =============================================================================
# ONE-SHOT COT
# =============================================================================
print('\nONE-SHOT COT')
oneShot_COT = extract_output_values_from_json_file('./dev/data/one-shot_COT_TF.out')
oneShot_COT = re_scan(tf_pattern, oneShot_COT)
scores(ground_truth, oneShot_COT)

# =============================================================================
# FEW-SHOT
# =============================================================================
print('\nFEW-SHOT')
fewShot = extract_output_values_from_json_file('./dev/data/few-shot.out')
fewShot = re_scan(tf_pattern, fewShot)
scores(ground_truth, fewShot)

# =============================================================================
# FEW SHOT COT
# =============================================================================
print('\nFEW-SHOT COT')
fewShot_COT = extract_output_values_from_json_file('./dev/data/few-shot_COT_TF.out')
fewShot_COT = re_scan(tf_pattern, fewShot_COT)
scores(ground_truth, fewShot_COT)
print() # blank line

# =============================================================================
# FEW-SHOT RAG
# =============================================================================
print('RAG no COT')
RAG_no_COT = extract_output_values_from_json_file('./dev/data/few-shot_RAG.out')
RAG_no_COT = re_scan(tf_pattern, RAG_no_COT)
scores(ground_truth, RAG_no_COT)


# =============================================================================
# FEW-SHOT RAG COT
# =============================================================================
print('\nRAG & COT')
RAG_COT= extract_output_values_from_json_file('./dev/data/few-shot_RAG_COT_TF.out')
RAG_COT = re_scan(tf_pattern, RAG_COT)
scores(ground_truth, RAG_COT)

# =============================================================================
# ENSEMBLE (BEST RESULTS)
# =============================================================================
import pandas as pd
df = pd.read_csv('./dev/data/dev_final.csv')
dev_final = df['baseline']
print('\nENSEMBLE')
scores(ground_truth, dev_final)
