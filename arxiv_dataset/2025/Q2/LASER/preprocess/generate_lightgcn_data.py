import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv

import numpy as np
from scipy.sparse import coo_matrix

from pre_utils import load_json, set_seed

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10

lm_hist_max = 30


def construct_mat(data, n_user, n_item):
    user_ids = data[:, 0].reshape(-1).tolist()
    item_ids = data[:, 1].reshape(-1).tolist()
    mat = coo_matrix((np.ones(len(user_ids)), (user_ids, item_ids)),
                     shape=[n_user, n_item])
    return mat


def generate_lightgcn_data(sequence_data):
    full_data = []
    max_uid, max_iid = 0, 0
    for uid, seq in sequence_data.items():
        item_seq, rating_seq = seq
        max_uid = max(max_uid, int(uid))
        for iid, lb in zip(item_seq, rating_seq):
            if lb > rating_threshold:
                full_data.append([int(uid), int(iid)])
            max_iid = max(max_iid, int(iid))
    random.shuffle(full_data)
    print(full_data[:5])
    full_data = np.array(full_data)
    max_uid += 1
    max_iid += 1
    print('data num', len(full_data), 'max uid', max_uid, 'max iid', max_iid)
    train_data = full_data[:int(len(full_data) * 0.9)]
    train_mat = construct_mat(train_data, max_uid, max_iid)
    test_data = full_data[int(len(full_data) * 0.9):]
    test_mat = construct_mat(test_data, max_uid, max_iid)
    return train_mat, test_mat


if __name__ == '__main__':
    set_seed(12345)

    DATA_DIR = '../data/'
    DATA_SET_NAME = 'amz-new'
    # DATA_SET_NAME = 'ml-10m-new'
    if DATA_SET_NAME in ['ml-10m-new']:
        rating_threshold = 3
    else:
        rating_threshold = 4
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')

    sequence_data = load_json(SEQUENCE_PATH)
    print('final loading data')

    print('generating data for light gcn')
    train_mat, test_mat = generate_lightgcn_data(sequence_data)
    # exit()
    with open(os.path.join(PROCESSED_DIR, 'trn_mat.pkl'), 'wb') as fs:
        pickle.dump(train_mat, fs)

    with open(os.path.join(PROCESSED_DIR, 'tst_mat.pkl'), 'wb') as fs:
        pickle.dump(test_mat, fs)

    with open(os.path.join(PROCESSED_DIR, 'val_mat.pkl'), 'wb') as fs:
        pickle.dump(test_mat, fs)
    print('Date saved')

