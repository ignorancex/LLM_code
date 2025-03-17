import torchvision
from torchvision import datasets

TIN_TRAIN_DIR = "/PATH_TO_DATA_DIR/TIMGNET/tiny-imagenet-200/restructured/train_restructured"
TIN_VAL_DIR = "/PATH_TO_DATA_DIR/TIMGNET/tiny-imagenet-200/restructured/val_restructured"

def create_train_test_set(data):
    if data.lower() == "cifar10":
        dataset = "CIFAR10"
        trainset = torchvision.datasets.CIFAR10(root='/PATH_TO_DATA_DIR/CIFAR10', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root='/PATH_TO_DATA_DIR/CIFAR10', train=False, download=True)
    elif data.lower() == "cifar100":
        dataset = "CIFAR100"
        trainset = torchvision.datasets.CIFAR100(root='/PATH_TO_DATA_DIR/CIFAR100', train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root='/PATH_TO_DATA_DIR/CIFAR100', train=False, download=True)
    elif data.lower() == "gtsrb":
        dataset = "GTSRB"
        trainset = torchvision.datasets.GTSRB(root='/PATH_TO_DATA_DIR/GTRSB', split='train', download=True)
        testset = torchvision.datasets.GTSRB(root='/PATH_TO_DATA_DIR/GTRSB', split='test', download=True)
    elif data.lower() == "stl10":
        dataset = "STL10"
        trainset = torchvision.datasets.STL10(root='/PATH_TO_DATA_DIR/STL10', split='train', download=True)
        testset = torchvision.datasets.STL10(root='/PATH_TO_DATA_DIR/STL10', split='test', download=True)
    elif data.lower() == "mnist":
        dataset = "MNIST"
        trainset = torchvision.datasets.MNIST(root='/PATH_TO_DATA_DIR/MNIST', train=True, download=True)
        testset = torchvision.datasets.MNIST(root='/PATH_TO_DATA_DIR/MNIST', train=False, download=True)
    elif data.lower() == "emnist":
        dataset = "EMNIST"
        trainset = torchvision.datasets.EMNIST(root='/PATH_TO_DATA_DIR/EMNIST', split='balanced', train=True, download=True)
        testset = torchvision.datasets.EMNIST(root='/PATH_TO_DATA_DIR/EMNIST', split='balanced', train=False, download=True)
    elif data.lower() == "fashionmnist" or data.lower() == "fashmnist":
        dataset = "FashionMNIST"
        trainset = torchvision.datasets.FashionMNIST(root='/PATH_TO_DATA_DIR/FashionMNIST', train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root='/PATH_TO_DATA_DIR/FashionMNIST', train=False, download=True)
    elif data.lower() == "svhn":
        dataset = "SVHN"
        trainset = torchvision.datasets.SVHN(root='/PATH_TO_DATA_DIR/SVHN', split='train', download=True)
        testset = torchvision.datasets.SVHN(root='/PATH_TO_DATA_DIR/SVHN', split='test', download=True)
    elif data.lower() == "food101":
        dataset = "Food101"
        trainset = torchvision.datasets.Food101(root='/PATH_TO_DATA_DIR/Food101', split='train', download=True)
        testset = torchvision.datasets.Food101(root='/PATH_TO_DATA_DIR/Food101', split='test', download=True)
    elif data.lower() == "tinyin" or data.lower() == "tinyimagenet":
        dataset = "TinyImageNet"
        trainset = datasets.ImageFolder(TIN_TRAIN_DIR)
        testset = datasets.ImageFolder(TIN_VAL_DIR)

    return trainset, testset, dataset

