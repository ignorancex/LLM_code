# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:54:34 2024

@author: Dan Schumacher
"""

# =============================================================================
# IMPORTS
# =============================================================================
from datasets import Dataset


import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# HOMEBREWED FUNCTIONS
import sys
sys.path.append('../../functions')
from load_data_dict import load_data_dict, load_test_dict
from calculations import create_embeddings
# =============================================================================
# LOAD DATA
# =============================================================================
# Our Dfs are dictionary objects but we need to turn them into Dataset Objects
df_dev_dict = load_data_dict('../data/dev.csv', separated=True)          # creates a structured dictionary from dev data
df_dev = Dataset.from_dict(df_dev_dict)                  # converts to Dataset Object
df_dev = df_dev.with_format("torch")                     # forces torch

df_train_dict = load_data_dict('../data/train.csv',separated=True)
df_train = Dataset.from_dict(df_train_dict)
df_train = df_train.with_format("torch")

df_test_dict = load_test_dict('../data/test.csv',separated=True)
df_test = Dataset.from_dict(df_test_dict)
df_test = df_test.with_format("torch")


# =============================================================================
# TEST EMBEDINGS
# =============================================================================

# SPLIT TRAIN INTO TRUE AND FALSE LABELS

# create pandas dataframe to access fancy indexing
pd_train = pd.DataFrame(df_train)

# Only Trues
pd_true = pd_train[pd_train['label'] == 1]
dict_true = pd_true.to_dict(orient='list')

# Only Falses
pd_false = pd_train[pd_train['label'] == 0]
dict_false = pd_false.to_dict(orient='list')

dev_embeddings = create_embeddings('nlpaueb/legal-bert-base-uncased', df_dev, batch_size=8)

true_train_embeddings = create_embeddings('nlpaueb/legal-bert-base-uncased', dict_true, batch_size=8)
false_train_embeddings = create_embeddings('nlpaueb/legal-bert-base-uncased', dict_false, batch_size=8)



# SPLIT TRAIN INTO TRUE AND FALSE LABELS
test_embeddings = create_embeddings('nlpaueb/legal-bert-base-uncased', df_test, batch_size=8)

# ITERATE
prompt_head_list = []
for test_cqa, test_emb in zip(df_test['cqa'], test_embeddings):
    test_emb = test_emb.reshape(1, -1)
    
    true_cosine_similarities = cosine_similarity(test_emb, true_train_embeddings)
    false_cosine_similarities = cosine_similarity(test_emb, false_train_embeddings)

    most_similar_true_index = true_cosine_similarities.argsort()[0][-1]
    most_similar_false_index = false_cosine_similarities.argsort()[0][-1]
    
    prompt_head_list.append(
        f"{dict_true['cqaal'][most_similar_true_index]}\n\n{dict_false['cqaal'][most_similar_false_index]}"
        )
    
# prompt_head_list[0]
    
# df_test_dict['prompt_head'] = prompt_head_list

# for idx, prompt_head, question in zip(df_test_dict['idx'], df_test_dict['prompt_head'], df_test_dict['question']):
#     if int(idx) <= 3:
#         print(
#             f'IDX:\n{idx}\n\nPROMPT HEAD:\n{prompt_head}\n\nCQA:\n{question}\n\n'
#             )
        
# SAVE OUTPUT AS JSON FILE

import json
json_file_path = '../../data/RAG_test.json'
with open(json_file_path, 'w', encoding='utf-8') as out_file:
    json.dump(df_test_dict, out_file)