from .dataset_info import get_classnames
import torch
from typing import Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def find_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std


def make_food101_loader(bs: int):
    TRAIN_PATH = "/home/cody/datasets/food-101/train/"
    VALID_PATH = "/home/cody/datasets/food-101/test/"

    food101_mean = [0.5567, 0.4381, 0.3198]
    food101_std = [0.2590, 0.2622, 0.2631]

    train_tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(food101_mean, food101_std)
            ])

    valid_tfms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(food101_mean, food101_std)
    ])
    train_ds_full = datasets.ImageFolder(root=TRAIN_PATH, transform=train_tfms)
    lengths = [int(len(train_ds_full)*.7), int(len(train_ds_full)*.3)]

    train_ds_holdout, train_ds = torch.utils.data.random_split(
                                    train_ds_full,
                                    lengths,
                                    generator=torch.Generator().manual_seed(42)
                                )

    valid_ds = datasets.ImageFolder(root=VALID_PATH, transform=valid_tfms)

    ss_train_dl = DataLoader(train_ds_holdout, batch_size=bs, shuffle=True,
                             num_workers=4, drop_last=True, pin_memory=True)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4,
                          drop_last=True, pin_memory=True)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {
            "ss_train": ss_train_dl,
            "train": train_dl,
            "val": valid_dl
    }

    dataset_sizes = {
            "ss_train": len(train_ds_holdout),
            "train": len(train_ds),
            "val": len(valid_ds)
    }

    return dataloaders, dataset_sizes


def make_imagenette_subset(bs: int, path: str, subset: list, normalize: bool):

    VALID_PATH = path + "val/"

    imagenette_mean = [0.4625, 0.4480, 0.4299]
    imagenette_std = [0.2863, 0.2828, 0.3051]

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
                transforms.Normalize(imagenette_mean,
                                     imagenette_std),
        )

    valid_tfms = transforms.Compose(transform_list)
    valid_ds = datasets.ImageFolder(root=VALID_PATH, transform=valid_tfms)
    valid_ds = Subset(valid_ds, subset)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {
            "val": valid_dl
    }

    dataset_sizes = {
            "val": len(valid_ds)
    }

    return dataloaders, dataset_sizes

def make_imagenet_loader(bs: int, path: str):

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    noramlize = transforms.Normalize(imagenet_mean, imagenet_std)
    
    train_tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                noramlize
            ])

    valid_tfms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                noramlize
    ])

    train_ds = datasets.ImageNet(root=path, split='train', transform=train_tfms)
    valid_ds = datasets.ImageNet(root=path, split='val', transform=valid_tfms)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4,
                          drop_last=True, pin_memory=True)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {
            "train": train_dl,
            "val": valid_dl
    }

    dataset_sizes = {
            "train": len(train_ds),
            "val": len(valid_ds)
    }
    
    class_names = train_ds.class_to_idx.keys()

    return dataloaders, dataset_sizes, class_names


def make_imagenette_loader(bs: int, path: str):
    TRAIN_PATH = path + '/train/'
    VALID_PATH = path + "/val/"

    imagenette_mean = [0.4625, 0.4480, 0.4299]
    imagenette_std = [0.2863, 0.2828, 0.3051]

    train_tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(imagenette_mean, imagenette_std)
            ])

    valid_tfms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenette_mean, imagenette_std)
    ])

    train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=train_tfms)
    valid_ds = datasets.ImageFolder(root=VALID_PATH, transform=valid_tfms)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4,
                          drop_last=True, pin_memory=True)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {
            "train": train_dl,
            "val": valid_dl
    }

    dataset_sizes = {
            "train": len(train_ds),
            "val": len(valid_ds)
    }

    class_names = get_classnames('imagenette')

    return dataloaders, dataset_sizes, class_names


def make_imagewoof_loader(bs: int, path: str):
    TRAIN_PATH = path + '/train/'
    VALID_PATH = path + "/val/"

    imagenette_mean = [0.4861, 0.4559, 0.3940]
    imagenette_std = [0.2603, 0.2531, 0.2619]

    train_tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(imagenette_mean, imagenette_std)
            ])

    valid_tfms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenette_mean, imagenette_std)
    ])

    train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=train_tfms)
    valid_ds = datasets.ImageFolder(root=VALID_PATH, transform=valid_tfms)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4,
                          drop_last=True, pin_memory=True)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {
            "train": train_dl,
            "val": valid_dl
    }

    dataset_sizes = {
            "train": len(train_ds),
            "val": len(valid_ds)
    }

    class_names = get_classnames('imagewoof')

    return dataloaders, dataset_sizes, class_names

def make_imagewoof_subset(bs: int, path: str, subset: list, normalize: bool):

    VALID_PATH = path + "val/"

    imagenette_mean = [0.4861, 0.4559, 0.3940]
    imagenette_std = [0.2603, 0.2531, 0.2619]

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
                transforms.Normalize(imagenette_mean,
                                     imagenette_std),
        )

    valid_tfms = transforms.Compose(transform_list)
    valid_ds = datasets.ImageFolder(root=VALID_PATH, transform=valid_tfms)
    valid_ds = Subset(valid_ds, subset)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {
            "val": valid_dl
    }

    dataset_sizes = {
            "val": len(valid_ds)
    }

    return dataloaders, dataset_sizes

def make_cifar10_data_loader(bs: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)

    valid_ds = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transform_test)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {"train": train_dl, "val": valid_dl}
    dataset_sizes = {"train": len(train_ds), "val": len(valid_ds)}
    class_names = train_ds.class_to_idx.keys()

    return dataloaders, dataset_sizes, class_names


def make_cifar10_subset(bs: int, subset: list, normalize: bool):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
        )
    transform_test = transforms.Compose(transform_list)

    valid_ds = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transform_test)

    valid_ds = Subset(valid_ds, subset)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {"val": valid_dl}
    dataset_sizes = {"val": len(valid_ds)}

    return dataloaders, dataset_sizes

def make_cifar100_data_loader(bs: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR100(root='./data', train=True, download=True,
                                transform=transform_train)

    valid_ds = datasets.CIFAR100(root='./data', train=False, download=True,
                                transform=transform_test)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    dataloaders = {"train": train_dl, "val": valid_dl}
    dataset_sizes = {"train": len(train_ds), "val": len(valid_ds)}
    class_names = train_ds.class_to_idx.keys()

    return dataloaders, dataset_sizes, class_names

def make_data_loader(bs, dataset='cifar10', subset: Optional[list] = None, normalize: bool = True):
    if dataset == 'cifar10':
        if subset is not None:
            return make_cifar10_subset(bs, subset, normalize)
        else:
            return make_cifar10_data_loader(bs)
    elif dataset == 'food101':
        return make_food101_loader(bs)
    elif dataset == 'imagenette':
        if subset is not None:
            return make_imagenette_subset(bs, path='./data/imagenette2-320/', subset=subset, normalize=normalize)
        return make_imagenette_loader(bs, path='./data/imagenette2-320/')
    elif dataset == 'imagewoof':
        if subset is not None:
            return make_imagewoof_subset(bs, path='./data/imagewoof2-320/', subset=subset, normalize=normalize)
        else:
            return make_imagewoof_loader(bs, path='./data/imagewoof2-320/')
    elif dataset == 'imagenet':
        return make_imagenet_loader(bs, path='./data/imagenet2012/')
    elif dataset == 'cifar100':
        return make_cifar100_data_loader(bs)
    else:
        print("dataset not valid")
