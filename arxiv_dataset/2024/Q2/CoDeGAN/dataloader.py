import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader


def CIFAR10(batch_size, root="./dataset/cifar10", train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=train, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    return loader


def FashionMNIST(batch_size, root="./dataset/fashionmnist", train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.FashionMNIST(
        root=root, train=train, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    return loader


class COIL20_dataset(Dataset):
    def __init__(self, root, train_class):
        datasets = np.load(root + "coil20.npz")
        imgs = datasets["images"]
        factors = datasets["labels"]
        num_class = train_class
        self.train_imgs = []
        self.train_factors = []
        for i in range(len(imgs)):
            self.train_imgs.append(np.squeeze(imgs[i]))
            self.train_factors.append(factors[i])
        del imgs
        del factors

        self.tranforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )

    def __getitem__(self, index):
        return self.tranforms(self.train_imgs[index]), self.train_factors[index]

    def __len__(self):
        return len(self.train_imgs)


def COIL20(batch_size, root="./dataset/coil20/", train_class=20):
    trainset = COIL20_dataset(root=root, train_class=train_class)
    loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    print("successfully loaded {} coil-20 data".format(len(trainset)))
    return loader


def get_dataloader(batch_size=10, dataset_name="cifar10", train=True):
    assert dataset_name in [
        "cifar10",
        "fashion",
        "coil20",
    ], "`dataset_name` must be one of the following values : `CIFAR10`, `FashionMNIST`, `COIL20`"
    if dataset_name == "cifar10":
        return CIFAR10(batch_size=batch_size, train=train)
    elif dataset_name == "fashion":
        return FashionMNIST(batch_size=batch_size, train=train)
    elif dataset_name == "coil20":
        return COIL20(batch_size=batch_size)
