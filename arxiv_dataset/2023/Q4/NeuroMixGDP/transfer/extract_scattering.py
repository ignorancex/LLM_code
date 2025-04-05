import argparse
import os

import torch
import numpy as np

from train_utils import get_device, train, test
from data import get_data, get_scatter_transform
from log import Logger


def get_scattered_data(loader, scattering, device, drop_last=False, sample_batches=False,name="train"):
    # pre-compute a scattering transform (if there is one) and return
    # a DataLoader

    scatters = []
    targets = []

    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        if scattering is not None:
            data = scattering(data)
        scatters.append(data)
        targets.append(target)

    scatters = torch.cat(scatters, axis=0)
    targets = torch.cat(targets, axis=0)

    np.save("transfer/features/scattering_mnist_{}.npy".format(name),scatters.cpu().numpy(),)

    data = torch.utils.data.TensorDataset(scatters, targets)
    return torch.utils.data.DataLoader(data,shuffle=False,
                                           num_workers=0, pin_memory=False)

def main(dataset, augment=False, batch_size=2048, mini_batch_size=256, sample_batches=False,
         lr=1, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=1, max_grad_norm=0.1,
         epochs=100, input_norm=None, num_groups=None, bn_noise_multiplier=None,
         max_epsilon=None, logdir=None):

    logger = Logger(logdir)
    device = get_device()

    train_data, test_data = get_data(dataset, augment=augment)
    scattering, K, (h, w) = get_scatter_transform(dataset)
    scattering.to(device)

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size

    # Batch accumulation and data augmentation with Poisson sampling isn't implemented
    if sample_batches:
        assert n_acc_steps == 1
        assert not augment

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)
    train_loader = get_scattered_data(train_loader, scattering, device,
                                            drop_last=True,
                                            sample_batches=sample_batches,name="train")
    test_loader = get_scattered_data(test_loader, scattering, device,name="test")


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'])
    parser.add_argument('--augment', action="store_true")
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default="SGD",
                        choices=["SGD", "Adam", "LR"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--input_norm', default=None,
                        choices=["GroupNorm", "BN"])
    parser.add_argument('--num_groups', type=int, default=81)
    parser.add_argument('--bn_noise_multiplier', type=float, default=6)
    parser.add_argument('--max_epsilon', type=float, default=None)
    parser.add_argument('--sample_batches', action="store_true")
    parser.add_argument('--logdir', default=None)
    args = parser.parse_args()
    main(**vars(args))
