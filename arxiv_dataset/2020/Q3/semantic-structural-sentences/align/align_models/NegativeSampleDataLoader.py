import torch
import pandas as pd


class NegativeSampleDataLoader(object):

    def __init__(self, _train_path, _valid_path, _batch_sz, _ns):
        self.train_path = _train_path
        self.valid_path = _valid_path
        self.batch_size = _batch_sz
        self.negative_samples = _ns
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

    def create_all_batches(self):
        permutation = torch.randperm(self.train_x.size()[0])
        batch_x = []
        batch_y_true = []
        batch_y_neg = []
        truth_labels = []
        # TODO: explore non-uniform negative sampling
        for i in range(0, self.train_x.size()[0], self.batch_size):
            indices = permutation[i:i + self.batch_size]
            if len(indices) == self.batch_size:  # hack, throwing out the non-mod batch size batches
                rem_ind = permutation[~indices]  # hack, gets a negative batch same size as positive batch size
                batch_x.append(self.train_x[indices])
                batch_y_true.append(self.train_y[indices])
                batch_y_neg.append(self.train_y[rem_ind])
                pos_labels = torch.tensor([1]*self.batch_size)
                neg_labels = torch.tensor([-1]*self.negative_samples)
                truth_labels.append(torch.cat([pos_labels, neg_labels]))
        return batch_x, batch_y_true, batch_y_neg, truth_labels
