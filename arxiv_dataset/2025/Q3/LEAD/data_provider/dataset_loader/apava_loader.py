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


class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])  # id of all samples
        val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
        test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = all_ids, train_ids, val_ids, test_ids

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
