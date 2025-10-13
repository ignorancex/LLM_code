import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import PIL

def get_imagenet_loaders(sample_level_train=1, sample_level_test=1, batch_size=256):
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_size = (224, 224)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = '/data/ILSVRC2012'#'/project/data_im'

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    #print('train data', len(train_data))
    #print('test data', len(test_data))
    train_loader, val_loader = create_data_loaders(sample_level_train, sample_level_test, train_data, test_data, batch_size)
    '''
    train_sampler = DistributedSampler(train_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = DataLoader(
            train_data,
            batch_size=batch_size, sampler=train_sampler, shuffle=False,
            num_workers=4, pin_memory=True)
    
    val_sampler = DistributedSampler(test_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    val_loader = DataLoader(
            test_data,
            batch_size=batch_size, sampler=val_sampler, shuffle=False,
            num_workers=4, pin_memory=True)
    '''
    return train_loader, val_loader

def create_data_loaders(sample_level_train, sample_level_test, train_data, test_data, batch_size):
    if sample_level_train < 1:
        rand_sampler_train = SubsetRandomSampler(torch.randperm(len(train_data))[:int(len(train_data) * sample_level_train)])

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size, sampler=rand_sampler_train, shuffle=False,
            num_workers=16, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size, shuffle=True,
            num_workers=16, pin_memory=True)

    if sample_level_test < 1:
        rand_sampler_test = SubsetRandomSampler(torch.randperm(len(test_data))[:int(len(test_data) * sample_level_test)])

        val_loader = DataLoader(
            test_data,
            batch_size=batch_size, sampler=rand_sampler_test, shuffle=False,
            num_workers=16, pin_memory=True)
    else:
        val_loader = DataLoader(
            test_data,
            batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True)

    return train_loader, val_loader
