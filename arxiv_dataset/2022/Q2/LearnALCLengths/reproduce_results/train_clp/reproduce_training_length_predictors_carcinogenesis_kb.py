import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import sys, os, json
from collections import Counter
import random

base_path = os.path.dirname(os.path.realpath(__file__)).split('reproduce_results')[0]
sys.path.append(base_path)

from helper_classes.experiment import Experiment
from sklearn.model_selection import train_test_split
from util.data import Data
from concept_length_predictors.models import RM
import json

data_path = base_path+"Datasets/carcinogenesis/Train_data/Data.json"
with open(data_path, "r") as file:
    data = json.load(file)
data = list(data.items())
data = Experiment.remove_minority_problem_types(data)
path_to_triples = base_path+"Datasets/carcinogenesis/Triples/"
triples = Data({"path_to_triples":path_to_triples})

as_classification = True
num_classes = max(v["target concept length"] for _,v in data)+1 if as_classification else 1
kwargs = {"learner_name":"GRU", "emb_model_name":"", "pretrained_embedding_path":base_path+"Datasets/carcinogenesis/Model_weights/ConEx_GRU.pt", "pretrained_length_learner":base_path+"Datasets/carcinogenesis/Model_weights/GRU.pt", "path_to_csv_embeddings":base_path+"Embeddings/carcinogenesis/ConEx_entity_embeddings.csv",
         "learning_rate":0.003, "decay_rate":0, "path_to_triples":path_to_triples,
         "random_seed":1, "embedding_dim":20, "num_entities":len(triples.entities),
          "num_relations":len(triples.relations), "num_ex":1000, "input_dropout":0.0, 
          "kernel_size":4, "num_of_output_channels":8, "feature_map_dropout":0.1,
          "hidden_dropout":0.1, "rnn_n_layers":2,'rnn_hidden':100, 'input_size':41,
          'linear_hidden':200, 'out_size':num_classes, 'dropout_prob': 0.1, 'num_units':500,
          'seed':10, 'seq_len':1000,'kernel_w':5, 'kernel_h':11, 'stride_w':1, 'stride_h':7,
          'conv_out':2040, 'mlp_n_layers':4, "as_classification":as_classification}

Models = ["GRU", "LSTM", "CNN", "MLP"]
print()
print('#'*50)
print('On Carcinogenesis knowledge base')
print('#'*50)
print()

random.seed(kwargs['seed'])

experiment = Experiment(kwargs)

data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)


print("*********************** Random Model ***********************")
Ytest = [v["target concept length"] for _,v in data_test]
data_t, data_v = train_test_split(data_train, test_size=0.1, random_state=123)
Yt = [v["target concept length"] for _,v in data_t]
Yv = [v["target concept length"] for _,v in data_v]
lengths = [v["target concept length"] for _,v in data]
max_len = max(lengths)
stats = Counter(lengths)
stats.update({l: 0 for l in range(1,max_len+1) if not l in stats})
stats = dict(sorted(stats.items(), key=lambda x: x[0]))
probs = list(map(lambda x: float(x)/len(lengths), stats.values()))
print("probs: ", {l+1 : s for l,s in enumerate(probs)})
print('\nResults on training, validation and test data...\n')
RM(Ytest=Yt, max_len=max_len, probs=probs)
RM(Ytest=Yv, max_len=max_len, probs=probs)
RM(Ytest=Ytest, max_len=max_len, probs=probs)
print("*********************** Random Model ***********************")

import argparse

parser = argparse.ArgumentParser()
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')
parser.add_argument('--final', type=str2bool, default=False, help="Whether to train on whole data and save model")
parser.add_argument('--test', type=str2bool, default=True, help="Whether to make predictions on test data")
parser.add_argument('--cross_validate', type=str2bool, default=True, help="Whether to use cross validation")
parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
parser.add_argument('--record_runtime', type=str2bool, default=True, help="Whether to record training runtime")
parser.add_argument('--save_model', type=str2bool, default=True, help="Whether to save the model after training")
args = parser.parse_args()
print()
if args.final:
    data_train = data
    args.test = False
    args.cross_validate = False
    args.record_runtime = True
    args.save_model = True
experiment.train_all_nets(Models, data_train, data_test, epochs=args.epochs, clp_batch_size=512, tc_batch_size=1024, kf_n_splits=10, cross_validate=args.cross_validate, test=args.test, save_model = args.save_model, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=args.record_runtime)
