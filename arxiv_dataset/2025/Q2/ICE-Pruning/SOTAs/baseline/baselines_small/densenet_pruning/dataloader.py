import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from torchvision import datasets, transforms


def get_tiny_imagenet_loaders(sample_level_train=1, sample_level_test=1, batch_size=256):
    data_dir = '/project/data' #'/home/wenhao/.cache/kagglehub/datasets/xiataokang/tinyimagenettorch/versions/1/tiny-imagenet-200'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader, val_loader = create_data_loaders(sample_level_train, sample_level_test, train_data, test_data, batch_size)

    return train_loader, val_loader

def get_cifar_10_loaders(sample_level=1, batch_size=128):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_data = datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
        transforms.ToTensor(), normalize]), download=True)

    test_data = datasets.CIFAR10(root='./cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(), normalize]))

    train_loader, val_loader = create_data_loaders(sample_level, train_data, test_data, batch_size)
    return train_loader, val_loader

def create_data_loaders(sample_level_train, sample_level_test, train_data, test_data, batch_size):
    if sample_level_train < 1:
        rand_sampler_train = SubsetRandomSampler(torch.randperm(len(train_data))[:int(len(train_data) * sample_level_train)])

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size, sampler=rand_sampler_train, shuffle=False,
            num_workers=0, pin_memory=True)

        rand_sampler_test = SubsetRandomSampler(torch.randperm(len(test_data))[:int(len(test_data) * sample_level_test)])

        val_loader = DataLoader(
            test_data,
            batch_size=128, sampler=rand_sampler_test, shuffle=False,
            num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True)

        val_loader = DataLoader(
            test_data,
            batch_size=128, shuffle=False,
            num_workers=0, pin_memory=True)

    return train_loader, val_loader
