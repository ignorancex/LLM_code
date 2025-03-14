import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import StratifiedKFold
import dgl

def collate(samples):
    # 'samples (graph, label)'
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

class GraphDataLoader():
    def __init__(self, dataset, batch_size, device,
                 collate_fn=collate, seed=0, shuffle=True,
                 split_name='fold10', fold_idx=0, split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if device >= 0 else {}

        labels = [l for _, l in dataset]

        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle
            )
        elif split_name == 'rand':
            train_idx, valid_idx = self._split_rand(
                labels, split_ratio, seed, shuffle
            )
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        self.train_loader = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

        self.valid_loader = DataLoader(
            dataset, sampler=valid_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        assert 0 <= fold_idx and fold_idx < 10, print(
            'fold_idx must be from 0 to 9.'
        )

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print(
            'train_set: test_set = %d : %d' % (len(train_idx), len(valid_idx))
        )
        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))

        np.random.seed(seed)
        np.random.shuffle(indices)

        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            'train_set: test_set = %d : %d' % (len(train_idx), len(valid_idx))
        )

        return train_idx, valid_idx

if __name__ == '__main__':
    from Temp.dataset import GINDataset
    dataset = GINDataset(name='PROTEINS', self_loop=True, degree_as_nlabel=False)

    Loader_list = []
    for idx in range(10):
        train_loader, valid_loader = GraphDataLoader(
            dataset, batch_size=128, device=0, collate_fn=collate,
            seed=9, shuffle=True, split_name='fold10', fold_idx=idx
        ).train_valid_loader()
        Loader_list.append((train_loader, valid_loader))
    print(Loader_list)
