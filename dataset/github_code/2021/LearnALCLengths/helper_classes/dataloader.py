from functools import singledispatchmethod
import numpy as np, random, torch, pandas as pd
from typing import List
from util.data import Data
from abc import ABCMeta
from tqdm import tqdm

class CLPDataLoader(Data, metaclass=ABCMeta):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.random_seed = kwargs['random_seed']
        random.seed(self.random_seed)

    @singledispatchmethod
    def load(self, embeddings, data, batch_size, shuffle, **kwargs):
        raise NotImplementedError
     
    @load.register  
    def _(self, embeddings: pd.DataFrame, data=None, batch_size=128, shuffle=True, **kwargs):
        if shuffle:
            random.shuffle(data)
        assert isinstance(data, List) or isinstance(data, np.ndarray), "Expected data type List or array, got object of type {}".format(type(data))
        datapoints = []
        targets = []
        for _, value in tqdm(data, desc="Loading data..."):
            pos = value['positive examples']
            neg = value['negative examples']
            try:
                datapoint_pos = torch.FloatTensor(embeddings.loc[pos].values)
                datapoint_neg = torch.FloatTensor(embeddings.loc[neg].values)
            except KeyError:
                try:
                    datapoint_pos = torch.FloatTensor(embeddings.loc[list(map(lambda x: x.replace("#", "."), pos))].values)
                    datapoint_neg = torch.FloatTensor(embeddings.loc[list(map(lambda x: x.replace("#", "."), neg))].values)
                except KeyError:
                    continue
            datapoint_pos = torch.cat([datapoint_pos, torch.ones((datapoint_pos.shape[0],1))], dim=1)
            datapoint_neg = torch.cat([datapoint_neg, -1. * torch.ones((datapoint_neg.shape[0],1))], dim=1)
            datapoint = torch.cat([datapoint_pos, datapoint_neg], dim=0)
            datapoints.append(datapoint.unsqueeze(0))
            try:
                targets.append(value["target concept length"])
            except KeyError:
                targets.append(0)
        return torch.cat(datapoints), torch.FloatTensor(targets)
            
    @load.register
    def _(self, embeddings: tuple, data=None, shuffle=True, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if shuffle:
            random.shuffle(data)
            
        if len(data) > batch_size:
            for i in tqdm(range(0, len(data)-batch_size+1, batch_size), desc="Loading data batches..."):
                datapoints = []
                targets = []
                data_ = data[i:i+batch_size]

                for _, value in data_:
                    pos = value['positive examples']
                    neg = value['negative examples']
                    #random.shuffle(pos)
                    try:
                        datapoint_pos = torch.cat([e(torch.tensor(list(map(lambda x: self.entity_to_idx[x], pos))).to(device)) for e in embeddings], 1)
                        datapoint_neg = torch.cat([e(torch.tensor(list(map(lambda x: self.entity_to_idx[x], neg))).to(device)) for e in embeddings], 1)
                    except KeyError:
                        continue
                    datapoint_pos = torch.cat([datapoint_pos, torch.ones((datapoint_pos.shape[0],1)).to(device)], dim=1)
                    datapoint_neg = torch.cat([datapoint_neg, (-1. * torch.ones((datapoint_neg.shape[0],1)).to(device))], dim=1)
                    datapoint = torch.cat([datapoint_pos, datapoint_neg], dim=0)
                    datapoints.append(datapoint.unsqueeze(0))
                    try:
                        targets.append(value["target concept length"])
                    except KeyError:
                        targets.append(0)
                yield torch.cat(datapoints), torch.FloatTensor(targets)
        else:
            datapoints = []
            targets = []
            for _, value in tqdm(data, desc="Loading data..."):
                pos = value['positive examples']
                neg = value['negative examples']
                #random.shuffle(pos)
                try:
                    datapoint_pos = torch.cat([e(torch.tensor(list(map(lambda x: self.entity_to_idx[x], pos))).to(device)) for e in embeddings], 1)
                    datapoint_neg = torch.cat([e(torch.tensor(list(map(lambda x: self.entity_to_idx[x], neg))).to(device)) for e in embeddings], 1)
                except KeyError:
                    continue
                datapoint_pos = torch.cat([datapoint_pos, torch.ones((datapoint_pos.shape[0],1)).to(device)], dim=1)
                datapoint_neg = torch.cat([datapoint_neg, (-1. * torch.ones((datapoint_neg.shape[0],1))).to(device)], dim=1)
                datapoint = torch.cat([datapoint_pos, datapoint_neg], dim=0)
                datapoints.append(datapoint.unsqueeze(0))
                try:
                    targets.append(value["target concept length"])
                except KeyError:
                    targets.append(0)
            yield torch.cat(datapoints), torch.FloatTensor(targets)
            
class HeadAndRelationBatchLoader(torch.utils.data.Dataset):
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        head_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.head_idx = head_rel_idx[:, 0]
        self.rel_idx = head_rel_idx[:, 1]
        self.tail_idx = list(er_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.tail_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.head_idx[idx], self.rel_idx[idx], y_vec
            
