import torch
import copy
import random
from collections import defaultdict


def dict_factory():
    return defaultdict(dict)

    
class CustomBatchSampler(torch.utils.data.sampler.Sampler[int]):
    def __init__(self, h5_dataset):
        self.h5_dataset = h5_dataset

    def __iter__(self):
        def custom_iterator():
            h5_dataset = copy.deepcopy(self.h5_dataset)
            ids = list(h5_dataset.h5_dict_tree.keys())
            
            
            for _ in range(len(h5_dataset)):
                id = random.choice(ids)
                ref = random.choice(list(h5_dataset.h5_dict_tree[id].keys()))
                h5_file = random.choice(list(h5_dataset.h5_dict_tree[id][ref].keys()))
                idx = h5_dataset.h5_dict_tree[id][ref][h5_file]['idx']
                yield idx

        return custom_iterator()
    
    def __len__(self):
        return len(self.h5_dataset)