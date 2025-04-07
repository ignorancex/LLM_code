import torch
import pandas as pd


class SupervisedDataLoader(object):

    def __init__(self, _train_path, _valid_path, _batch_sz):
        self.train_path = _train_path
        self.valid_path = _valid_path
        self.batch_size = _batch_sz
        self.train_x, self.train_y, self.valid_x, self.valid_y = self.process()

    def process(self):
        train_df = pd.read_csv(self.train_path)
        train_df_x = train_df[['em1_idx', 'rel_idx', 'em2_idx']]
        train_df_y = train_df[['sent_id']]
        train_x = torch.from_numpy(train_df_x.values)
        train_y = torch.from_numpy(train_df_y.values)
        valid_df = pd.read_csv(self.valid_path)
        valid_df_x = valid_df[['em1_idx', 'rel_idx', 'em2_idx']]
        valid_df_y = valid_df[['sent_id']]
        valid_x = torch.from_numpy(valid_df_x.values)
        valid_y = torch.from_numpy(valid_df_y.values)
        return train_x, train_y, valid_x, valid_y

    def get_validation(self):
        return self.valid_x, self.valid_y

    def get_validation_batches(self):
        valid_batch_x = []
        valid_batch_y = []
        permutation = torch.randperm(self.valid_x.size()[0])
        for i in range(0, self.valid_x.size()[0], self.batch_size):
            indices = permutation[i:i + self.batch_size]
            valid_batch_x.append(self.valid_x[indices])
            valid_batch_y.append(self.valid_y[indices])
        return valid_batch_x, valid_batch_y

    def create_all_batches(self):
        permutation = torch.randperm(self.train_x.size()[0])
        batch_x = []
        batch_y = []
        for i in range(0, self.train_x.size()[0], self.batch_size):
            indices = permutation[i:i + self.batch_size]
            batch_x.append(self.train_x[indices])
            batch_y.append(self.train_y[indices])
        return batch_x, batch_y
