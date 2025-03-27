"""
Models in cifar are based on https://github.com/kuangliu/pytorch-cifar, and models of mnist are modified on them.
Models in imagenet comes from torchvision 0.8.1.

Model family:
mnist: input 28x28, 1 channel
cifar: input 32x32, 3 channels
imagenet: input 224x224, 3 channels
"""

from . import cifar, mnist

def get_family(dataset):
    if dataset in ["cifar10", "cifar100", "svhn", "gtsrb"]:
        return "cifar"
    if dataset in ["mnist", "kmnist", "fashionmnist"]:
        return "mnist"
    return ""


REGISTRY_MODEL = {
    "cifar": {
        "resnet18": cifar.resnet18,
        "resnet34": cifar.resnet34,
        "resnet50": cifar.resnet50,
        "lenet5": cifar.lenet5,
        "wrn16_4": cifar.wrn16_4,
        "wrn28_10": cifar.wrn28_10,
        "vgg16": cifar.vgg16,
    },
    "mnist": {
        "resnet18": mnist.resnet18,
        "resnet34": mnist.resnet34,
        "resnet50": mnist.resnet50,
        "lenet5": mnist.lenet5,
        "wrn16_4": mnist.wrn16_4,
        "wrn28_10": mnist.wrn28_10,
        "vgg16": mnist.vgg16,
    }
}

