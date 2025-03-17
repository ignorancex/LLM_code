from utils import *
import argparse
import warnings
import pickle
import numpy as np
import os
import torch
import pandas as pd
from tqdm import tqdm
from calibration_utils import *

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description='Binning setup')
parser.add_argument('predict_probs_pkl', type=str,
                    help='.pickle file containing softmax prediction probabilities')
parser.add_argument('predicted_labels', type=str,
                    help='.pickle file containing predicted labels')
parser.add_argument('labels', type=str,
                    help='.pickle file containing actual labels')
parser.add_argument('bins', type=int,
                    help='Number of bins to calculate calibration error')

args = parser.parse_args()

predict_probs_pkl = args.predict_probs_pkl
predicted_labels = args.predicted_labels
labels = args.labels
n_bins = args.bins

# -------------------------------------------------------------------------------------------------------------------#
# analysis of predictions

ece_list = []
fce_list = []
uf_list = []
of_list = []
bins_list = []
ece_breakdown = []
fce_breakdown = []

print("Binning probabilities...")

with open(predict_probs_pkl, 'rb') as handle:
    soft_preds = pickle.load(handle)
with open(predicted_labels, 'rb') as handle:
    preds = pickle.load(handle)
with open(labels, 'rb') as handle:
    labels = pickle.load(handle)


incorrect = []
correct = []
for i in range(len(preds)):
    if preds[i] != labels[i]:
        incorrect.append([torch.softmax(soft_preds[i], 0), int(preds[i]), int(labels[i])])
    else:
        correct.append([torch.softmax(soft_preds[i], 0), int(preds[i]), int(labels[i])])

# calculate overconfidence --> expectation of confidence over incorrect predictions
of = []
for i in range(len(incorrect)):
    of.append(max(incorrect[i][0]))

# calculate underconfidence --> expectation of 1-confidence over correct predictions
uf = []
for i in range(len(correct)):
    uf.append(1 - max(correct[i][0]))

uf_list.append(np.mean(uf))
of_list.append(np.mean(of))

ece_dict = {'soft_preds': np.array([np.array(torch.softmax(x, 0)) for x in soft_preds]),
            'preds': [int(x) for x in preds],
            'labels': [int(x) for x in labels]}

print("Calculating ECE...")
ece_vals, ece = expected_calibration_error(np.array(ece_dict['labels']),
                                           ece_dict['soft_preds'], num_bins=n_bins)

bins_list.append(n_bins)
ece_list.append(ece)
ece_breakdown.append(ece_vals)
print("ECE calculations done!")

print("Calculating FCE...")
fce_vals, fce = fuzzy_calibration_error(np.array(ece_dict['labels']), ece_dict['soft_preds'], n_bins)

fce_list.append(fce)
fce_breakdown.append(fce_vals)
print("FCE calculations done!")

                
print ("ECE: %.3f \nFCE: %.3f  " %(ece, fce))

