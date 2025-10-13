"""Cifar100 utilities

To run this, you will need the cifar100 dataset locally under
`dataset/cifar100.` This dataset can be downloaded from 
https://www.cs.toronto.edu/~kriz/cifar.html
"""

import jax
import numpy as np
from pathlib import Path
import pickle

import sys
sys.path.append('../')
from common import *

import flaxmodels as fm

cache = {}

def load_data(preprocess_cnn=True, seed=None, actv_layer='relu5_3', sub_samp=1, sub_samp_key=None, normalize=True):
    global cache

    if seed is None:
        seed = new_seed()

    parent_path = Path(__file__).parent / 'dataset'
    dataset_path = parent_path / 'cifar100'

    with open(dataset_path / 'train', 'rb') as fp:
        train_set = pickle.load(fp, encoding='bytes')

    with open(dataset_path / 'test', 'rb') as fp:
        test_set = pickle.load(fp, encoding='bytes')

    with open(dataset_path / 'meta', 'rb') as fp:
        meta_set = pickle.load(fp)


    train_labs = train_set[b'fine_labels']
    train_data = train_set[b'data']

    test_labs = test_set[b'fine_labels']
    test_data = test_set[b'data']

    all_labs = train_labs + test_labs
    all_data = np.concatenate((train_data, test_data), axis=0)

    if normalize:
        all_data = (all_data - all_data.mean()) / all_data.std()

    if sub_samp < 1:
        data_len = all_data.shape[0]
        sub_len = np.round(sub_samp * data_len).astype(int)

        ss_rng = np.random.default_rng(sub_samp_key)
        sub_samp_idx = ss_rng.choice(data_len, size=sub_len, replace=False)

        all_data = all_data[sub_samp_idx]
        all_labs = np.array(all_labs)[sub_samp_idx]
        print(f'info: sub samp data to size={all_data.shape[0]}')

    if preprocess_cnn and actv_layer != 'id':
        if actv_layer in cache:
            all_data = cache[actv_layer]
            print(f'info: using cache for layer={actv_layer}')
        else:
            all_data = all_data.reshape(-1, 32, 32, 3, order='F')

            model = fm.VGG16(pretrained='imagenet', normalize=False, include_head=False, output='activations')
            params =model.init(jax.random.PRNGKey(0), all_data[[0]])

            batches = []
            apply_fn = lambda b: model.apply(params, b, train=False)
            apply_fn = jax.jit(apply_fn)

            for b in tqdm(np.split(all_data, 1000)):
                out = apply_fn(b)
                batches.append(out[actv_layer])

            all_data = np.concatenate(batches, axis=0)
            all_data = all_data.reshape(all_data.shape[0], -1)

            cache[actv_layer] = all_data

    label_names = meta_set['fine_label_names']

    return {
        'data': all_data,
        'labels': all_labs,
        'names': label_names
    }

