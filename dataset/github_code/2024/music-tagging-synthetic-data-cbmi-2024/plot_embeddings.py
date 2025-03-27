import argparse
import os
import random
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
from music_tagger import N_FRAMES
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

GENRES = ['Pop', 'Hip-Hop', 'Classical']
NUM_EXAMPLES = 50
LABELS = {
    0: 'Pop real',
    1: 'Pop synthetic',
    2: 'HipHop real',
    3: 'HipHop synthetic',
    4: 'Jazz real',
    5: 'Jazz synthetic'
}

examples = []

for k, genre in enumerate(GENRES):
    genre_real_folder = os.path.join(TARGET_EMBEDDINGS_DIR, genre)
    genre_real_files = [os.path.join(genre_real_folder, x) for x in os.listdir(genre_real_folder) if x.endswith('.p')]
    random.shuffle(genre_real_files)
    genre_real_files = genre_real_files[:NUM_EXAMPLES]
    genre_real_examples = [pickle.load(open(x, 'rb')) for x in genre_real_files]
    genre_real_examples = [x[:N_FRAMES, :] for x in genre_real_examples]
    examples.extend(genre_real_examples)
    genre_fake_folder = os.path.join(SOURCE_EMBEDDINGS_DIR, genre)
    genre_fake_files = [os.path.join(genre_fake_folder, x) for x in os.listdir(genre_fake_folder) if x.endswith('.p')]
    random.shuffle(genre_fake_files)
    genre_fake_files = genre_fake_files[:NUM_EXAMPLES]
    genre_fake_examples = [pickle.load(open(x, 'rb')) for x in genre_fake_files]
    genre_fake_examples = [x[:N_FRAMES, : ] for x in genre_fake_examples]
    examples.extend(genre_fake_examples)

examples = np.asarray(examples)


both_base_model = tf.keras.models.load_model('models/mcnn_both_split_0.h5')
both_model = both_base_model.layers[-2]

da_base_model = tf.keras.models.load_model('models/mcnn_da_split_0.h5',
                                           custom_objects={"contrastive_loss": tfa.losses.contrastive_loss})
da_model = da_base_model.layers[3]

both_examples = both_model.predict(examples)
da_examples = da_model.predict(examples)

c = ['b', 'b', 'r', 'r', 'k', 'k']
m = ["^", "o", "^", "o", "^", "o"]

now = datetime.datetime.now()
date_str = now.strftime("%d-%m-%Y_%H:%M:%S")

tsne = TSNE(n_iter=50000, early_exaggeration=120.0, perplexity=10.0)
both_coord = tsne.fit_transform(both_examples)
l = []
for g in LABELS.keys():
    plt.scatter(both_coord[g * NUM_EXAMPLES:(g + 1) * NUM_EXAMPLES, 0],
                both_coord[g * NUM_EXAMPLES:(g + 1) * NUM_EXAMPLES, 1],
                edgecolors=c[g],
                marker=m[g],
                facecolors='none')
    l.append(LABELS[g])
plt.legend(l)
plt.title('without domain adaptation')
plt.savefig(os.path.join(results_dir, 'without_da_' + date_str + '.png'))
plt.figure()
da_coord = tsne.fit_transform(da_examples)
for g in LABELS.keys():
    plt.scatter(da_coord[g * NUM_EXAMPLES:(g + 1) * NUM_EXAMPLES, 0],
                da_coord[g * NUM_EXAMPLES:(g + 1) * NUM_EXAMPLES, 1],
                edgecolors=c[g],
                marker=m[g],
                facecolors='none')
plt.title('with domain adaptation')
plt.legend(l)
plt.savefig(os.path.join(results_dir, 'with_da_' + date_str + '.png'))
plt.show()



