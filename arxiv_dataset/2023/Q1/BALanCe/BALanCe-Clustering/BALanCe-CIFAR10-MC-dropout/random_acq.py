import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils
import random

def acquire(pool, B=10):
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    origin_pos_lt = []
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            origin_pos_lt += list(pos.data.cpu().numpy())

    pos_lt = random.sample(origin_pos_lt, k=B)

    return pos_lt


