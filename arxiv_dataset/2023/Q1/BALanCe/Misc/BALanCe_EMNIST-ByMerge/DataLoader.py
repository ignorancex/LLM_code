import torch
import torch.utils.data
import numpy as np
import csv
import random
from Buffer import Buffer
from Pool import Pool
import collections

from torchvision import datasets, transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, buff, num_class=47):
        if isinstance(buff, Buffer):
            self.features = buff.features
            self.labels = buff.labels
        if isinstance(buff, Pool):
            self.features = buff.features
            self.labels = np.expand_dims(buff.labels, axis=1)
        
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)
        
        self.num_class = num_class
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __len__(self):
        return self.labels.shape[0]

    def get_weight(self):
        if len(self.labels) == 1:
            count = collections.Counter(list(self.labels[0]))
        else:
            count = collections.Counter(np.squeeze(self.labels))
        #num_class = len(count)
        weight_lt = []
        for i in range(self.num_class):
            if i not in count or count[i] == 0:
                weight = 0.0
            else:
                weight = 1./count[i]
            weight_lt.append(weight)
        weight_lt = np.array(weight_lt)
        return weight_lt

    def __getitem__(self, idx):
        #c_id = self.ids[idx]
        feat = self.features[idx]
        #feat = torch.Tensor(feat)
        feat = self.transform(feat)

        label = torch.LongTensor(self.labels[idx])
        return (feat, label)



