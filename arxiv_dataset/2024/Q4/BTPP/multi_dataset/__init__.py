import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .mnist import MyMNIST
from .cifar10 import MyCIFAR10

def get_dataset(args, rank=0):
    if args.datasets == 'mnist':
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = MyMNIST(rank=rank, world_size=args.nodes, data_hete=args.data_hete, 
                                       root = './data', train=True, download=True, transform=trans_train)
        
    elif args.datasets == 'cifar10':
        trans_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = MyCIFAR10(rank=rank, world_size=args.nodes, data_hete=args.data_hete, 
                                       root = './data', train=True, download=True, transform=trans_train)

    generator = torch.Generator().manual_seed(rank)

    dataloader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          sampler=torch.utils.data.RandomSampler(
                              data_source=trainset,
                              replacement=True,
                              num_samples=len(trainset),
                              generator = generator
                          )
                          )
    return dataloader
    
def get_evaluate_datasets(args):
    if args.datasets == 'mnist':
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset_loss = datasets.MNIST('./data', train=True, download=True, transform=trans_train)
        testset = datasets.MNIST('./data', train=False, download=True, transform=trans_train)

    elif args.datasets == 'cifar10':
        trans_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset_loss = datasets.CIFAR10(root = './data', train=True, download=True, transform=trans_train)
        testset = datasets.CIFAR10(root = './data', train=False, download=True, transform=trans_test)

    dataloader_loss = DataLoader(trainset_loss, batch_size=64, num_workers=2, shuffle=False)
    testloader = DataLoader(testset, batch_size=64, num_workers=2, shuffle=False)

    return dataloader_loss, testloader

def get_warm_up_datasets(args):
    if args.datasets == 'mnist':
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST('./data', train=True, download=True, transform=trans_train)
        
    elif args.datasets == 'cifar10':
        trans_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])

        trainset = datasets.CIFAR10(root = './data', train=True, download=True, transform=trans_train)

    generator = torch.Generator().manual_seed(0)

    dataloader = DataLoader(dataset=trainset,
                          batch_size=32,
                          sampler=torch.utils.data.RandomSampler(
                              data_source=trainset,
                              replacement=True,
                              num_samples=len(trainset),
                              generator = generator
                          )
                          )
    return dataloader

def get_dataset_shuffle(args, rank=0):
    if args.datasets == 'mnist':
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = MyMNIST(rank=rank, world_size=args.nodes, data_hete=args.data_hete, 
                                       root = './data', train=True, download=True, transform=trans_train)
        
    elif args.datasets == 'cifar10':
        trans_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = MyCIFAR10(rank=rank, world_size=args.nodes, data_hete=args.data_hete, 
                                       root = './data', train=True, download=True, transform=trans_train)

    generator = torch.Generator().manual_seed(rank)

    dataloader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True
                          )
    return dataloader