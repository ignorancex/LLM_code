"""
CelebA Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more
"""
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CelebA(Dataset):
    def __init__(self, data_dir, split, transform):
        self.data_dir = data_dir
        split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.split = split

        dtypes = defaultdict(lambda: np.int32)
        dtypes['File_Name'] = 'str'
        self.metadata_df = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.txt'), sep=' ', header=0,
                                       index_col=False, dtype=dtypes)
        if split in ['train', 'val', 'test']:
            split_values = pd.read_csv(os.path.join(data_dir, 'list_eval_partition.txt'), header=None, index_col=False,
                                       sep=' ', names=['file', 'split']).values[:, 1].astype(int)
            self.split_idx = split_values == split_dict[split]
        elif split == 'all':
            self.split_idx = np.full(len(self.metadata_df), True)
        self.metadata_df = self.metadata_df[self.split_idx]

        # Get the y values
        self.y_array = self.metadata_df.values[:, 1:].astype(int)

        # Extract filenames and splits
        self.filename_array = self.metadata_df['File_Name'].values

        self.targets = torch.tensor(self.y_array)

        self.transform = transform

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'img_align_celeba', self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)

        y = self.targets[idx]
        img_filename = self.filename_array[idx]
        return x, y, img_filename


def get_transform_celeba():
    return transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
