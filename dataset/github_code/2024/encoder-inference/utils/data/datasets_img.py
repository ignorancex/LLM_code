from typing import Iterable, Callable
import torch, torchvision
from datasets import load_dataset

from .utils import TORCH_DATA_PATH, EmbedDataset


class Dataset:
    def __init__(self, x: Iterable, y: Iterable, transform: Callable = None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if self.transform is not None:
            x = self.transform(Image.fromarray(x))

        return x, y

    def __len__(self):
        return len(self.x)


def CIFAR10(root=TORCH_DATA_PATH, train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)

def CIFAR100(root=TORCH_DATA_PATH, train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)

def SVHN(root=TORCH_DATA_PATH, train=True, transform=None):
    return torchvision.datasets.SVHN(root=root, split='train' if train else 'test',
                        transform=transform, download=True)

def STL10(root=TORCH_DATA_PATH, train=True, transform=None):
    return torchvision.datasets.STL10(root=root, split='train' if train else 'test',
                        transform=transform, download=True)

class Food101:
    def __init__(self, train=True, transform=None):
        data = load_dataset("food101")
        self.data = data["train" if train else "validation"]
        self.train = train
        self.trans = transform

    def __getitem__(self, idx):
        x = self.data[idx]["image"].convert('RGB')
        y = self.data[idx]["label"]
        if self.trans is not None:
            x = self.trans(x)
        return x, y

    def __len__(self):
        return len(self.data)


__dataset_zoo__ = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "stl10": STL10,
    "food101": Food101,
}


def get_img_dataset(name: str, train=True, trans=None):
    return __dataset_zoo__[name](train=train, transform=trans)

def get_img_embed_dataset(name: str, encoder, train=True):
    dataset = __dataset_zoo__[name](train=train)
    return EmbedDataset.build(dataset, encoder)
