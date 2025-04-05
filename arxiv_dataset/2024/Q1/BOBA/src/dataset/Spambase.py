import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Spambase(Dataset):
    """
    Spambase Dataset
    """

    def __init__(self, root, train=True, download=True):
        """
        :param root: root of the dataset
        :param train: load the training set / testing set
        :param download:
        """

        self.root = os.path.join(root, 'uci')
        self.path = os.path.join(root, 'uci', 'spambase.data')
        self.train = train

        if download:
            self.download()

        X_train, Y_train, X_test, Y_test = self.load_and_preprocess()

        self.classes = ['0 - not_spam', '1 - spam']
        if train:
            self.data = X_train
            self.targets = Y_train
        else:
            self.data = X_test
            self.targets = Y_test

    def download(self):
        if os.path.exists(self.path):
            return

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        from urllib.request import urlretrieve
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
        print('Downloading Spambase dataset ...')
        urlretrieve(url, self.path)

    def load_and_preprocess(self):
        mat = pd.read_csv(self.path, header=None)

        # train test split
        train_indices = list(filter(lambda x: x % 5 != 0, range(len(mat))))
        test_indices = list(filter(lambda x: x % 5 == 0, range(len(mat))))
        mat_train = mat.iloc[train_indices, :]
        mat_test = mat.iloc[test_indices, :]

        X_train = torch.Tensor(np.array(mat_train.iloc[:, :-1]))
        Y_train = torch.LongTensor(np.array(mat_train.iloc[:, -1]))

        X_test = torch.Tensor(np.array(mat_test.iloc[:, :-1]))
        Y_test = torch.LongTensor(np.array(mat_test.iloc[:, -1]))

        # normalize
        X_mean = X_train.mean(dim=0)
        X_std = X_train.std(dim=0)

        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        return X_train, Y_train, X_test, Y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_dir = '~/data'
    data_dir = os.path.expanduser(data_dir)
    spam = Spambase(root=data_dir, download=True)
    print(len(spam))
    loader = DataLoader(spam, batch_size=32, shuffle=False)
    for i, (X, Y) in enumerate(loader):
        print(X.shape, Y.shape)
