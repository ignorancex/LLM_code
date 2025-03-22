import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
    load_data_by_ids,
)
import warnings
import random

warnings.filterwarnings('ignore')


def get_id_list_reeg_baca(args, label_path, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for REEG-BACA data
    Args:
        args: arguments
        label_path: directory of label.npy file
        a: ratio of ids in training set
        b: ratio of ids in training and validation set
    Returns:
        all_ids: list of all IDs
        train_ids: list of IDs for training set
        val_ids: list of IDs for validation set
        test_ids: list of IDs for test set
    '''
    # random shuffle to break the potential influence of human named ID order,
    # e.g., put all healthy subjects first or put subjects with more samples first, etc.
    # (which could cause data imbalance in training, validation, and test sets)
    data_list = np.load(label_path)

    all_ids = list(data_list[:, 1])
    if args.cross_val == 'fixed':  # fixed split
        random.seed(42)
    elif args.cross_val == 'mccv':  # Monte Carlo cross-validation
        random.seed(args.seed)
    else:
        raise ValueError('Invalid cross_val. Please use fixed or mccv.')
    random.shuffle(all_ids)
    train_ids = all_ids[:int(a * len(all_ids))]
    val_ids = all_ids[int(a * len(all_ids)):int(b * len(all_ids))]
    test_ids = all_ids[int(b * len(all_ids)):]

    return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)


class REEGBACALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        a, b = 0.6, 0.8
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_reeg_baca(args, self.label_path, a, b)
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or ALL.')

        self.X, self.y = load_data_by_ids(self.data_path, self.label_path, ids)
        self.X = bandpass_filter_func(self.X, fs=args.sampling_rate, lowcut=args.low_cut, highcut=args.high_cut)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
