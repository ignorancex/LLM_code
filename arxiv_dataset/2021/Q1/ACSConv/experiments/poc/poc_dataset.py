import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class BaseDatasetShape(Dataset):
    def __init__(self, data_root, num_data):
        self.data_root = data_root
        self.len = num_data        
    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_root,'shape_{}.npz'.format(index)))
        shape = data['shape']
        segs = data['segs']
        return torch.from_numpy(shape).float().unsqueeze(0), torch.from_numpy(segs.astype(float)).float()
    def __len__(self):
        return self.len
        
class BaseDatasetVoxel(Dataset):
    def __init__(self, data_root, num_data):
        self.data_root = data_root
        self.len = num_data        
    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_root,'voxel_{}.npz'.format(index)))
        voxel = data['voxel']
        segs = data['segs']
        return torch.from_numpy(voxel).float().unsqueeze(0), torch.from_numpy(segs.astype(float)).float()
    def __len__(self):
        return self.len