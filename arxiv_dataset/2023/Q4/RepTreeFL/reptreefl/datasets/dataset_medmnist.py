import torch.utils.data as data
import torchvision.transforms as transforms
import random
from torch.utils.data import Subset
import numpy as np
import medmnist
from medmnist import INFO
import constants

def read_all_files_medmnist():
    info = INFO[constants.MEDMNIST_DATASET]
    DataClass = getattr(medmnist, info['python_class'])

    return DataClass

def get_data_loaders(DataClass):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    download = True

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=constants.BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=constants.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_datasets(DataClass):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    download = True

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    return train_dataset, val_dataset, test_dataset

def get_subset_dataset(dataset, split="train", balanced=False):
    label_counts = np.bincount(dataset.labels[:, 0])
    if balanced:
        label_distribution = [1. / len(label_counts) for i in range(len(label_counts))]
        label_distribution = np.array(label_distribution)
    else:
        label_distribution = label_counts / len(dataset.labels)

    desired_subset_size = 0 # no. samples in total
    if split == "train":
        desired_subset_size = constants.DESIRED_DATASET_SUBSET_SIZE
    elif split == "val":
        desired_subset_size = 100
    elif split == "test":
        desired_subset_size = 200

    if balanced:
        class_subset_sizes = (label_distribution * desired_subset_size).astype(int)
    else:
        class_subset_sizes = (label_distribution * desired_subset_size).astype(int)

    selected_samples = []
    # Select samples from each class
    for target_label in range(len(label_counts)):
        # Get the number of samples with current target label
        # Create an empty list to store the indices
        indices = []

        # Iterate over the dataset and check if the label matches the target label
        for i in range(len(dataset)):
            _, label = dataset[i]  # Assuming each item in the dataset is a tuple of (data, label)
            if label == target_label:
                indices.append(i)
        random.shuffle(indices)
        selected_samples.extend(indices[:class_subset_sizes[target_label]])

    random.shuffle(selected_samples)

    # Create a new subset dataset
    subset_dataset = Subset(dataset, selected_samples)

    return subset_dataset

def get_subset_dataset_split(train_dataset, val_dataset, test_dataset, balanced=False):
    subset_train_dataset = get_subset_dataset(train_dataset, split="train", balanced=balanced)
    subset_val_dataset = get_subset_dataset(val_dataset, split="val", balanced=balanced)
    subset_test_dataset = get_subset_dataset(test_dataset, split="test", balanced=balanced)

    return subset_train_dataset, subset_val_dataset, subset_test_dataset

def get_medmnist_dataset(balanced=False):
    data_class = read_all_files_medmnist()
    train_dataset, val_dataset, test_dataset = get_datasets(data_class)

    subset_train_dataset, subset_val_dataset, subset_test_dataset = get_subset_dataset_split(train_dataset, val_dataset, test_dataset, balanced=balanced)

    return subset_train_dataset, subset_val_dataset, subset_test_dataset