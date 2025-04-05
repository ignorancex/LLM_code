import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os

from .distortions import distortions, test_distortions


def make_cifar10_c(partition_idxs, severity, data_dir='../data'):
    """

    """
    data_dir = os.path.join(data_dir, 'torchvision')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    cifar_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        distortion = random_distortion()
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            # import matplotlib.pyplot as plt
            x, y = dataset[sid]

            # plt.imshow(x)
            # plt.show()

            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            # plt.imshow(x.permute((1, 2, 0)))
            # plt.show()
            cifar_c[sid] = x
            labels[sid] = y

    assert None not in cifar_c
    assert None not in labels

    cifar_c = torch.stack(cifar_c)
    labels = torch.LongTensor(labels)

    return cifar_c, labels, corruption


def random_distortion():
    selected_distortions = distortions
    num_distortions = len(selected_distortions)
    i = np.random.randint(num_distortions)
    return selected_distortions[i]


def test():
    partition_idxs = {
        0: [*range(1)],
        1: [*range(1, 2)]
    }
    make_cifar10_c(partition_idxs)