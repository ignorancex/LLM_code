import os
import random

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

# constant for CelebA
CELEBA_MALE_INDEX = 20


def load_naive_dataset(dataset_name: str, batch_size: int = 128):
    # root directory
    root = f"datasets/{dataset_name}/" if dataset_name in ["CIFAR10", "STL10"] else "datasets/"
    os.makedirs(root, exist_ok=True)

    # transform
    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
            ]
        )
    else:  # CIFAR10, STL10, or CelebA
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    # dataset
    if dataset_name == "MNIST":
        train = datasets.MNIST(root=root, download=True, train=True, transform=transform)
        test = datasets.MNIST(root=root, download=True, train=False, transform=transform)
    elif dataset_name == "CIFAR10":
        train = datasets.CIFAR10(root=root, download=True, train=True, transform=transform)
        test = datasets.CIFAR10(root=root, download=True, train=False, transform=transform)
    elif dataset_name == "STL10":
        train = datasets.STL10(root=root, split="train", download=True, transform=transform)
        test = datasets.STL10(root=root, split="test", download=True, transform=transform)
        train, test = torch.utils.data.random_split(ConcatDataset([train, test]), [12000, 1000])
    else:  # CelebA

        def target_transform(target):
            return target[CELEBA_MALE_INDEX]

        train = datasets.CelebA(
            root=root,
            split="train",
            target_type="attr",
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        test = datasets.CelebA(
            root=root,
            split="test",
            target_type="attr",
            transform=transform,
            target_transform=target_transform,
            download=True,
        )

    # dataloader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    return train_loader, test_loader


def load(
    dataset_name: str,
    batch_size: int,
    normal_class: tuple[int, ...],
    n_unlabeled_normal: int,
    n_labeled_sensitive: int,
    n_unlabeled_sensitive: int,
    n_test: int,
    is_normalize: bool = True,
) -> tuple[DataLoader, DataLoader]:
    # root directory
    root = f"datasets/{dataset_name}/" if dataset_name in ["CIFAR10", "STL10"] else "datasets/"
    os.makedirs(root, exist_ok=True)

    # transform
    if dataset_name == "MNIST":
        base = [
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
        ]
    elif dataset_name == "CelebA":
        base = [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ]
    else:  # CIFAR10 or STL10
        base = [transforms.ToTensor()]

    if is_normalize:
        normalizer = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
        normalizer = []

    train_transform = transforms.Compose(base + normalizer)
    test_transform = transforms.Compose(base)

    # dataset
    if dataset_name == "MNIST":
        train = datasets.MNIST(root=root, download=True, train=True, transform=train_transform)
        test = datasets.MNIST(root=root, download=True, train=False, transform=test_transform)
    elif dataset_name == "CIFAR10":
        train = datasets.CIFAR10(root=root, download=True, train=True, transform=train_transform)
        test = datasets.CIFAR10(root=root, download=True, train=False, transform=test_transform)
    elif dataset_name == "STL10":
        train = datasets.STL10(root=root, split="train", download=True, transform=train_transform)
        test = datasets.STL10(root=root, split="test", download=True, transform=test_transform)
    else:  # CelebA

        def target_transform(target):
            return target[CELEBA_MALE_INDEX]

        train = datasets.CelebA(
            root=root,
            split="train",
            target_type="attr",
            transform=train_transform,
            target_transform=target_transform,
            download=True,
        )
        test = datasets.CelebA(
            root=root,
            split="test",
            target_type="attr",
            transform=test_transform,
            target_transform=target_transform,
            download=True,
        )

    # Train
    if dataset_name == "STL10":
        train_indices = train.labels
    elif dataset_name == "CelebA":
        train_indices = train.attr[:, CELEBA_MALE_INDEX]
    else:
        train_indices = train.targets

    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices)

    train_is_normal = torch.isin(train_indices, torch.tensor(normal_class))
    train_normal_indices = torch.where(train_is_normal)[0].tolist()
    train_sensitive_indices = torch.where(~train_is_normal)[0].tolist()

    train_normal_bag = random.sample(train_normal_indices, k=n_unlabeled_normal)
    train_sensitive_bag = random.sample(train_sensitive_indices, k=n_labeled_sensitive + n_unlabeled_sensitive)

    train_positive_bag = train_sensitive_bag[:n_labeled_sensitive]
    train_unlabeled_bag = train_normal_bag + train_sensitive_bag[n_labeled_sensitive:]

    for i in train_positive_bag:
        if dataset_name == "STL10":
            train.labels[i] = 1
        elif dataset_name == "CelebA":
            train.attr[i, CELEBA_MALE_INDEX] = 1
        else:
            train.targets[i] = 1

    for i in train_unlabeled_bag:
        if dataset_name == "STL10":
            train.labels[i] = 0
        elif dataset_name == "CelebA":
            train.attr[i, CELEBA_MALE_INDEX] = 0
        else:
            train.targets[i] = 0

    train_subset = Subset(train, train_positive_bag + train_unlabeled_bag)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Test
    if dataset_name == "STL10":
        test_indices = test.labels
    elif dataset_name == "CelebA":
        test_indices = test.attr[:, CELEBA_MALE_INDEX]
    else:
        test_indices = test.targets

    if not torch.is_tensor(test_indices):
        test_indices = torch.tensor(test_indices)

    test_is_normal = torch.isin(test_indices, torch.tensor(normal_class))
    test_normal_indices = torch.where(test_is_normal)[0].tolist()
    test_normal_bag = test_normal_indices[:n_test]
    # test_normal_bag = random.sample(test_normal_indices, k=min(len(test_normal_indices), n_test))

    for i in test_normal_bag:
        if dataset_name == "STL10":
            test.labels[i] = 0
        elif dataset_name == "CelebA":
            test.attr[i, CELEBA_MALE_INDEX] = 0
        else:
            test.targets[i] = 0

    test_subset = Subset(test, test_normal_bag)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def to_tensor(data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    data = []
    label = []
    for batch in data_loader:
        data.append(batch[0])
        label.append(batch[1])

    return torch.cat(data, dim=0), torch.cat(label, dim=0)
