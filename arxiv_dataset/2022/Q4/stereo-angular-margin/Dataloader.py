from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision import datasets
from torchvision.datasets import CIFAR100, CIFAR10
import torch


def Loadcifar10(batchsize=256):

    BATCH_SIZE = batchsize

    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    train_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_data = CIFAR10(download=True, root="./data",
                         transform=train_transform)
    test_data = CIFAR10(root="./data", train=False, transform=test_transform)

    # Specify The Batch size
    train_dl = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_data, BATCH_SIZE, shuffle=True)

    return train_dl, test_dl


def Loadcifar100(batchsize=256):

    BATCH_SIZE = batchsize

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_data = CIFAR100(download=True, root="./data",
                          transform=train_transform)
    test_data = CIFAR100(root="./data", train=False, transform=test_transform)

    # Specify The Batch size
    train_dl = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_data, BATCH_SIZE, shuffle=True)

    return train_dl, test_dl


def Loadmalariadata(batchsize=256):

    BATCH_SIZE = batchsize

    stats = ((0.5507, 0.4580, 0.4950), (0.3110, 0.2617, 0.2791))

    train_transform = tt.Compose([tt.Resize([128,128]),
                                  #tt.RandomCrop(128),
                                  # tt.RandomHorizontalFlip(),
                                  # tt.RandomVerticalFlip(),
                                  # tt.RandomRotation(45),
                                  tt.ToTensor(),
                                  tt.Normalize(*stats)])

    test_transform = tt.Compose([tt.Resize([128,128]),
                                 #tt.RandomCrop(128),
                                 tt.ToTensor(),
                                 tt.Normalize(*stats)])

    train_dataset = datasets.ImageFolder(
        'malariadataset/train', transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    test_dataset = datasets.ImageFolder(
        'malariadataset/test', transform=test_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    return train_dataloader, test_dataloader


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, sim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std
