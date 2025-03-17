from functools import partial
from torchvision import datasets, transforms
from .imagenet1k import ImageNet1k

train_transform_cifar = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform_cifar = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_transform_mnist = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

test_transform_mnist = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


REGISTRY_TRAIN_DATASET = {
    "cifar10": partial(datasets.CIFAR10, train=True, transform=train_transform_cifar),
    "cifar100": partial(datasets.CIFAR100, train=True, transform=train_transform_cifar),
    "svhn": partial(datasets.SVHN, split="train", transform=train_transform_cifar),
    "imagenet_tiny": partial(ImageNet1k, train=True, transform=train_transform_cifar),
    "fashionmnist": partial(datasets.FashionMNIST, train=True, transform=train_transform_mnist),
    "mnist": partial(datasets.MNIST, train=True, transform=train_transform_mnist),
    "kmnist": partial(datasets.KMNIST, train=True, transform=train_transform_mnist),
    "emnist": partial(datasets.EMNIST, split="digits", train=True, transform=train_transform_mnist),
    "emnistletters": partial(datasets.EMNIST, split="letters", train=True, transform=train_transform_mnist),
}

REGISTRY_QUERY_DATASET = {
    "cifar10": partial(datasets.CIFAR10, train=True, transform=test_transform_cifar),
    "cifar100": partial(datasets.CIFAR100, train=True, transform=test_transform_cifar),
    "svhn": partial(datasets.SVHN, split="train", transform=test_transform_cifar),
    "imagenet_tiny": partial(ImageNet1k, train=True, transform=test_transform_cifar),
    "fashionmnist": partial(datasets.FashionMNIST, train=True, transform=test_transform_mnist),
    "mnist": partial(datasets.MNIST, train=True, transform=test_transform_mnist),
    "kmnist": partial(datasets.KMNIST, train=True, transform=test_transform_mnist),
    "emnist": partial(datasets.EMNIST, split="digits", train=True, transform=test_transform_mnist),
    "emnistletters": partial(datasets.EMNIST, split="letters", train=True, transform=test_transform_mnist),
}

REGISTRY_TEST_DATASET = {
    "cifar10": partial(datasets.CIFAR10, train=False, transform=test_transform_cifar),
    "cifar100": partial(datasets.CIFAR100, train=False, transform=test_transform_cifar),
    "svhn": partial(datasets.SVHN, split="test", transform=test_transform_cifar),
    "imagenet_tiny": partial(ImageNet1k, train=False, transform=train_transform_cifar),
    "fashionmnist": partial(datasets.FashionMNIST, train=False, transform=test_transform_mnist),
    "mnist": partial(datasets.MNIST, train=False, transform=test_transform_mnist),
    "kmnist": partial(datasets.KMNIST, train=False, transform=test_transform_mnist),
    "emnist": partial(datasets.EMNIST, split="digits", train=False, transform=test_transform_mnist),
    "emnistletters": partial(datasets.EMNIST, split="letters", train=False, transform=test_transform_mnist),
}

REGISTRY_INFO = {
    "cifar10": [10, 32, 3],
    "cifar100": [100, 32, 3],
    "svhn": [10, 32, 3],
    "imagenet_tiny": [200, 32, 3],
    "fashionmnist": [10, 28, 1],
    "mnist": [10, 28, 1],
    "kmnist": [10, 28, 1],
    "emnist": [10, 28, 1],
    "emnistletters": [10, 28, 1],
}
