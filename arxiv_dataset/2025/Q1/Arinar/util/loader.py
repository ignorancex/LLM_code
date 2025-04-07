import os
import time

import numpy as np
import io
import json
import h5py

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target
    
    
class CachedH5FolderDev(Dataset):
    def __init__(self, root):
        self.h5_path = os.path.join(root, "mar_cache.h5")
        self.file = None

        h5_json = os.path.join(root, "mar_cache_h5.json")

        with open(h5_json, "r") as f:
            cache = json.load(f)

        label_names = sorted(list(set([elem.split("/")[0] for elem in cache])))
        self.label2idx = {label: i for i, label in enumerate(label_names)}
        self.samples = {}
        for c in cache:
            self.samples[c] = self.label2idx[c.split("/")[0]]
        self.samples = list(self.samples.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        path, target = self.samples[index]

        data = self.file[path]
        data = np.load(io.BytesIO(np.array(data)))

        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target
    

class CachedNpzData(Dataset):
    def __init__(self, root):
        super().__init__()
        print("Loading data into memory...")
        start = time.time()
        saved_npz = np.load(os.path.join(root, "mar_cache.npz"))
        self.labels = saved_npz['label']
        self.data = saved_npz['data']  # (num of samples, 2 hflips, 32, 16, 16)
        print(f"Loaded {len(self.labels)} samples in {time.time() - start:.2f}s")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        target = self.labels[index]
        data = self.data[index]

        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data[0]
        else:
            moments = data[1]

        return moments, target
