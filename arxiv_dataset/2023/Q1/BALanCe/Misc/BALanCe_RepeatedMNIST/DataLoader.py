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
    def __init__(self, buff):
        if isinstance(buff, Buffer):
            self.features = buff.features
            self.labels = buff.labels
        if isinstance(buff, Pool):
            self.features = buff.features
            self.labels = np.expand_dims(buff.labels, axis=1)
            #print(self.labels.shape)
            #print(self.features.shape)
        #print(f'Dataset: features size: {self.features.shape}, label size: {self.labels.shape}') 
        
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __len__(self):
        return self.labels.shape[0]

    def get_weight(self):
        count = collections.Counter(np.squeeze(self.labels))
        num_class = len(count)
        weight_lt = []
        for i in range(num_class):
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



