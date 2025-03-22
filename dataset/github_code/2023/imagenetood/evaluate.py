import os
import pickle
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".") 
import numpy as np
import torch
import argparse


length = 13

in_label = [[] for _ in range(length)]
out_label = [[] for _ in range(length)]
scores = {}
parser = argparse.ArgumentParser(description="", allow_abbrev=False)
parser.add_argument("--in_pkl", type=str, default=True)
parser.add_argument("--out_pkl", type=str, required=True)
args = parser.parse_args()

with open(args.in_pkl, 'rb') as f:
    whole_data = pickle.load(f)
    for i in range(length):
        data = whole_data[i]
        in_label[i] += data['label']
        for key in data.keys():
            if key == 'label':
                continue
            if key not in scores:
                scores[key] = [{'in': [], 'out': []} for _ in range(length)]
            scores[key][i]['in'] += list(data[key])

with open(args.out_pkl, 'rb') as f:
    whole_data = pickle.load(f)
    for i in range(length):
        data = whole_data[i]
        out_label[i] += data['label']
        for key in data.keys():
            if key == 'label':
                continue
            scores[key][i]['out'] += list(data[key])
print("AUROC New Class Detection")
for key in scores.keys():
    print(f"{key} Score: {np.mean([roc_auc_score([1]*len(in_label[i]) + [0]*len(out_label[i]), scores[key][i]['in'] + scores[key][i]['out'] , average=None) for i in range(length)])}")
print("AUROC Failure Detection")
for key in scores.keys():
    print(f"{key} Score: {np.mean([roc_auc_score(in_label[i] + out_label[i], scores[key][i]['in'] + scores[key][i]['out'], sample_weight=[1/len(in_label[i])]*len(in_label[i]) + [1/len(out_label[i])]*len(out_label[i]) , average=None) for i in range(length)])}")