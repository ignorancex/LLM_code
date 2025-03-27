import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import Dataset


class MSTARDataset(Dataset):
    def __init__(self, path, is_train=True):
        self.is_train = is_train

        self.images = []
        self.labels = []
        self.serial_number = []

        if is_train:
            self.transform = transforms.Compose([RandomCrop(88), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([CenterCrop(88), transforms.ToTensor()])
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx]
        _label = self.labels[idx]
        _serial_number = self.serial_number[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label, _serial_number

    def _load_data(self, path):
        mode = 'train' if self.is_train else 'test'

        image_list = glob.glob(os.path.join(path, f'{mode}/*/*.npy'))
        label_list = glob.glob(os.path.join(path, f'{mode}/*/*.json'))
        image_list = sorted(image_list, key=os.path.basename)
        label_list = sorted(label_list, key=os.path.basename)

        for image_path, label_path in tqdm.tqdm(zip(image_list, label_list), desc=f'load {mode} data set'):
            self.images.append(np.load(image_path))

            with open(label_path, mode='r', encoding='utf-8') as f:
                _label = json.load(f)

            self.labels.append(_label['class_id'])
            self.serial_number.append(_label['serial_number'])
        

class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=2)

        h, w, _ = _input.shape
        oh, ow = self.size

        dh = h - oh
        dw = w - ow
        y = np.random.randint(0, dh) if dh > 0 else 0
        x = np.random.randint(0, dw) if dw > 0 else 0
        oh = oh if dh > 0 else h
        ow = ow if dw > 0 else w

        return _input[y: y + oh, x: x + ow, :]
    

class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=2)

        h, w, _ = _input.shape
        oh, ow = self.size
        y = (h - oh) // 2
        x = (w - ow) // 2

        return _input[y: y + oh, x: x + ow, :]