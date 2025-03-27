import os
import torch
import torchvision
import torchvision.transforms as transforms

'''
data loader for ImageNet dataset
'''
def imagenet_data_loader(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_batch_size = args.batch_size
    val_batch_size = args.eval_batch_size
    pin_memory = True
    workers = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        )

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


'''
data loader for CIFAR-10 dataset
'''
def cifar10_dataset_loader(args):
    traindir = args.data_dir
    valdir = args.data_dir
    train_batch_size = args.batch_size
    val_batch_size = args.eval_batch_size

    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
         ])

    transform_val = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
         ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=traindir, train=True, download=True, transform=transform_train)

    val_dataset = torchvision.datasets.CIFAR10(
        root=valdir, train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True, num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=args.workers)

    return train_loader, val_loader

'''
data loader for CIFAR-10 dataset
'''
def cifar100_dataset_loader(args):
    traindir = args.data_dir
    valdir = args.data_dir
    train_batch_size = args.batch_size
    val_batch_size = args.eval_batch_size

    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
         ])

    transform_val = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
         ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=traindir, train=True, download=True, transform=transform_train)

    val_dataset = torchvision.datasets.CIFAR100(
        root=valdir, train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True, num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=args.workers)

    return train_loader, val_loader
