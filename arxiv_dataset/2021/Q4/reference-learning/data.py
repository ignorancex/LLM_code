import torch
import torchvision
import numpy as np

tr = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor()]
        )

def load_FMNIST():
    train = torchvision.datasets.FashionMNIST('./data/FMNIST',
                                               train=True,
                                               download=True,
                                               transform=tr)
    test = torchvision.datasets.FashionMNIST('./data/FMNIST',
                                              train=False,
                                              download=True,
                                              transform=tr)
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=len(train))
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=len(test))
    x_train = iter(train_loader).next()[0]
    x_test = iter(test_loader).next()[0]
    return x_train, x_test


def load_EMNIST():
    train = torchvision.datasets.EMNIST('./data/EMNIST',
                                        train=True,
                                        download=True,
                                        split='letters',
                                        transform=tr)
    test = torchvision.datasets.EMNIST('./data/EMNIST',
                                       train=False,
                                       download=True,
                                       split='letters',
                                       transform=tr)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=len(train))
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=len(test))
    x_train = iter(train_loader).next()[0]
    x_test = iter(test_loader).next()[0]
    return x_train, x_test


def load_CIFAR10():
    CIFAR10_train = torchvision.datasets.CIFAR10('./data/CIFAR',
                                                 train=True,
                                                 download=True)
    CIFAR10_test = torchvision.datasets.CIFAR10('./data/CIFAR',
                                                train=False,
                                                download=True)

    rgb2gray = torch.tensor([[0.2989], [0.5870], [0.1140]])
    x_train = torch.from_numpy(CIFAR10_train.data.astype(np.float32))@rgb2gray
    x_test = torch.from_numpy(CIFAR10_test.data.astype(np.float32))@rgb2gray

    x_train = (x_train/255.0)[:, None,:, :, 0]
    x_test = (x_test/255.0)[:, None,:, :, 0]
    return x_train, x_test


def load_MNIST():
    train = torchvision.datasets.MNIST('./data/MNIST',
                                       train=True,
                                       download=True,
                                       transform=tr)
    test = torchvision.datasets.MNIST('./data/MNIST',
                                      train=False,
                                      download=True,
                                      transform=tr)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=len(train))
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=len(test))
    x_train = iter(train_loader).next()[0]
    x_test = iter(test_loader).next()[0]
    return x_train, x_test


def load_SVHN():
    train = torchvision.datasets.SVHN('./data/SVHN', split='train', download=True)
    test = torchvision.datasets.SVHN('./data/SVHN', split='test', download=True)
    x_train = np.moveaxis(train.data, 1, -1)
    x_test = np.moveaxis(test.data, 1, -1)

    rgb2gray = torch.tensor([[0.2989], [0.5870], [0.1140]])
    x_train = torch.from_numpy(x_train.astype(np.float32))@rgb2gray
    x_test = torch.from_numpy(x_test.astype(np.float32))@rgb2gray

    x_train = (x_train/255.0)[:, None,:, :, 0]
    x_test = (x_test/255.0)[:, None,:, :, 0]
    return x_train, x_test