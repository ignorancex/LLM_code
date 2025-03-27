import argparse
import os
import random
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from music_tagger import N_FRAMES, N_MELS, GENRES, collect_feature_data
import tensorflow_addons as tfa
import datetime


parser = argparse.ArgumentParser(description='Plot embeddings with and without domain adaptation.')
parser.add_argument('--source-data',
                    type=str,
                    required=True,
                    help='Path to the source (human-synthetic) embeddings directory')
parser.add_argument('--target-data',
                    type=str,
                    required=True,
                    help='Path to the target (real) embeddings directory')


args = parser.parse_args()

SOURCE_EMBEDDINGS_DIR = args.source_data
TARGET_EMBEDDINGS_DIR = args.target_data

results_dir = 'results'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

trg_only_model = tf.keras.models.load_model('models/mcnn_both_split_0.h5')
_, _, val_tracks = collect_feature_data(SOURCE_EMBEDDINGS_DIR, TARGET_EMBEDDINGS_DIR, split_idx=0)

X = np.zeros((len(val_tracks), N_FRAMES, N_MELS))
y_true = np.zeros((len(val_tracks), len(GENRES)))
for k, track in enumerate(val_tracks):
    data = pickle.load(open(track['path'], 'rb'))
    y_true[k, track['label']] = 1
    X[k, :, :] = data[:N_FRAMES, :]

y_pred = trg_only_model.predict(X)
matrix = confusion_matrix(y_true, y_pred, labels=GENRES)
print(matrix)

