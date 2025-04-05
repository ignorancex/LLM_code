import pickle as pkl
import pandas as pd
import numpy as np

np.random.seed(35)

import argparse
parser = argparse.ArgumentParser()

## Required parameters
#bert-base-multilingual-cased, xlm-roberta-base, bert-base-cased
#bert, xlmroberta

parser.add_argument("--model_path", default='bert-base-cased', type=str, help="name or path to checkpoint")
parser.add_argument("--model_type", default='bert', type=str, help="type of embedding")
parser.add_argument("--add_data", default='0', type=str, help="activate by iteration number if you want to add fresh labeled data from previous iteration")
parser.add_argument("--num_epochs", default=1, type=int, help="number of epochs the model will train for")
arg = parser.parse_args()

print('----------------Iteration', arg.add_data, '------------------------')

if arg.add_data != '0':
    with open('../data/train_data_'+str(int(arg.add_data)-1)+'.pkl', 'rb') as pkl_in:
         train_data = pkl.load(pkl_in)
    df_add = pd.read_csv('../data/sample2label'+arg.add_data+'.csv')
    add_data = [[text, label] for text, label in zip(df_add['Text'].tolist(), df_add['Label'].tolist())]
    train_data = train_data + add_data
    with open('../data/train_data_'+arg.add_data+'.pkl', 'wb') as pkl_out:
         pkl.dump(train_data, pkl_out)
else:
    with open('../data/train_data_0.pkl', 'rb') as pkl_in:
         train_data = pkl.load(pkl_in)

train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
train_df.head()

with open('../data/unlabelled_data_'+arg.add_data+'.pkl', 'rb') as pkl_in: 
    X_pool = pkl.load(pkl_in)
print('X_pool length..', len(X_pool))

from simpletransformers.classification import ClassificationModel

model_args = {"manual_seed": 35,
             "num_train_epochs": arg.num_epochs,
             "output_dir": "outputs_red_entropy_args/", 
             "overwrite_output_dir": True,
             "do_lower_case": False,
             "early_stopping_delta": 0.01,
             "max_seq_length": 512,
             "dataloader_num_workers": 0,
             "train_batch_size": 8, 
             "eval_batch_size": 8}

model = ClassificationModel(arg.model_type, arg.model_path, num_labels=3, args=model_args)

#do active learning -> teach the model
model.train_model(train_df)

#evaluate trained model on held out annotated data - 100 samples
import nltk

with open('../data/test_data.pkl', 'rb') as pkl_in:
    test_data = pkl.load(pkl_in)
test_df = pd.DataFrame(test_data)
test_df.columns = ["text", "labels"]
#print(test_df.head())

import sklearn
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.classification_report)
print('-----RESULT------', result)

#predict
predictions, raw_outputs = model.predict(X_pool)

from scipy.special import softmax
from scipy.stats import entropy

def uncertainity(raw_op, x, query = 50, mode = 'entropy'):
    prob = np.array([softmax(arr) for arr in raw_op])
    assert np.array(raw_op).shape == prob.shape
    if mode == 'entropy':
        uncertain = entropy(prob.T)
    else:
        uncertain = [1 - max(arr) for arr in prob]
    uncertain = [uncertain[i] *-1 for i in range(len(uncertain))] 
    ranks = list(np.argsort(uncertain))
    return [[x[ranks.index(i)]] for i in range(query)], [ranks.index(i) for i in range(query)]
    #returns the text and the indices that need to be removed from the unlabelled pool


# ...obtaining new labels from the Oracle..
#print(raw_outputs[:5])

x_getlabel, indices = uncertainity(raw_outputs, X_pool) #uncertainity sampling

import csv

# field names  
fields = ['Text']
with open('../data/sample2label'+str(int(arg.add_data)+1)+'.csv', 'w') as f:  
    
    write = csv.writer(f)  
    write.writerow(fields) 
    write.writerows(x_getlabel)


#remove these instances from the pool
X_pool = np.delete(X_pool, indices, axis=0)

with open('../data/unlabelled_data_'+str(int(arg.add_data)+1)+'.pkl', 'wb') as pkl_out:  #instead of cs was new
    pkl.dump(X_pool, pkl_out)
  
#run code again
