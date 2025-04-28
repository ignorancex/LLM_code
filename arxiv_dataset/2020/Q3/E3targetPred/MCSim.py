## All required packages
## Scientific Packakges
import sys, os, re, gc, time
import numpy as np
import pandas as pd
from random import sample
from scipy.io import savemat
from collections import Counter
import matplotlib.pyplot as plt

## Keras
from keras import metrics
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

## Sklearn-metrics
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score

## Custom functions
from models import E3_LSE
from utils import *
from feature_encoding import * ## Feature encoding utils
from perf_metrics import * ## Some of hand crafted performance metrics


## Loading and pre-processing E3-Target dataset
data_path = 'https://raw.githubusercontent.com/psychemistz/Colab_temp/master/E3target_pred/1metadata.csv'
dataset = pd.read_csv(data_path, index_col=None)

## Extract features
E3_features, Sub_features, Pair_features = Extract_Features(dataset, gap_size=6)
target_label = to_categorical(dataset['Label'])

"""LVs = Number of Latent Variables
   Gaps = Number of Gaps in CKSAAP Feature
   lmd = Weight of Reconstruction Loss (Lambda)
   num_Trials = Number of Trials for each LV, Gap combination at given Lambda value. 
"""
LVs = np.asarray(range(2,6))
Gaps = np.asarray(range(0,9))
lmd = 0.5 
num_Trials = 20

MCSim(LVs, Gaps, lmd, num_Trials, E3_features, Sub_features, Pair_features, target_label)