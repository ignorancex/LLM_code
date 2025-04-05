import os
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms


def create_torchvision_dataset(dataset_name='mnist', data_dir='../data', cache=True):
    """
    Use dataset given in torchvision
    :param dataset_name: name of the dataset
    :param data_dir: directory of the dataset, e.g., ../data
    :param cache: load the dataset to memory to speed up training
    :return: train_dataset, test_dataset
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean and std of mnist
        ])

        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)


    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    else:
        raise NotImplementedError('It is not a torchvision dataset! Please check the dataset name. ')

    if cache:
        train_dataset = dataset_in_memory(train_dataset)
        test_dataset = dataset_in_memory(test_dataset)

    return train_dataset, test_dataset


def dataset_in_memory(dataset, device=None):
    X = []
    Y = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x)
        Y.append(y)

    X = torch.stack(X)
    Y = torch.LongTensor(Y)

    if device is not None:
        X = X.to(device)
        Y = Y.to(device)

    return TensorDataset(X, Y)
