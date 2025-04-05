import pickle as pkl
import pandas as pd
import numpy as np

np.random.seed(35)

from fuzzywuzzy import fuzz
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()

## Required parameters
#bert-base-multilingual-cased, xlm-roberta-base, bert-base-cased
#bert, xlmroberta
parser.add_argument("--model_path", default='bert-base-cased', type=str, help="name or path to checkpoint")
parser.add_argument("--model_type", default='bert', type=str, help="type of embedding")

arg = parser.parse_args()

with open('../data/train_data_rel.pkl', 'rb') as pkl_in:
     train_data = pkl.load(pkl_in)

train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
train_df = train_df.sample(frac=1, random_state=30).reset_index(drop=True)

print('Length of Total Data........', len(train_df))
print(train_df.head())

from simpletransformers.classification import ClassificationModel


model_args = {"manual_seed": 35,
             "num_train_epochs": 5,
             "output_dir": "outputs_rel/",
             "overwrite_output_dir": True,
             "do_lower_case": False,
             "dataloader_num_workers": 0,
             "max_seq_length": 512,
             "evaluate_during_training": True,
             "logging_steps": 100,
             "train_batch_size": 8, #with 4 it's worse
             "eval_batch_size": 8}

model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=model_args)

msk = np.random.rand(len(train_df)) < 0.8
eval_df = train_df[~msk]
train_df = train_df[msk]

print('Length of Training Data........', len(train_df))
print('Length of Evaluation Data........', len(eval_df))

model.train_model(train_df, eval_df=eval_df)

#optional part below
#evaluate trained model on held out annotated data - 100 samples
import nltk

with open('../data/test_data_rel.pkl', 'rb') as pkl_in:
    test_data = pkl.load(pkl_in)

test_df = pd.DataFrame(test_data)
test_df.columns = ["text", "labels"]

print('Length of Testing Data........', len(test_df))

import sklearn
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.classification_report)
print('-----RESULT------', result)
