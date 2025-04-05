import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import skimage as sk


class NoisyDataset(Dataset):
    """
    A dataset which adds random noise to images
    """

    def __init__(self, clean_dataset, noise, severity, args):

        self.X = []
        self.Y = []

        for x, y in clean_dataset:
            self.X.append(x.numpy())
            self.Y.append(y)

        self.X = np.stack(self.X)

        print(self.X.shape)

        print(self.X.min(), self.X.max())

        if args.dataset == 'mnist':
            self.X = self.X * 0.3081 + 0.1307  # normalize to [0, 1]

        elif args.dataset == 'cifar10':
            print(self.X.shape)
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
            std = np.array([0.2470, 0.2435, 0.2616]).reshape((3, 1, 1))
            self.X = self.X * std + mean
            # print('haha', self.X.min(), self.X.max())

        else:
            raise NotImplementedError

        self.X = np.clip(self.X, 0, 1)
        self.X = self.X * 255.0  # normalize to [0, 255]

        noise_func = {
            'gaussian': gaussian_noise,
            'shot': shot_noise,
            'impulse': impulse_noise,
            'speckle': speckle_noise,
        }[noise]

        # print(self.X.min(), self.X.max())

        self.NoisyX = noise_func(self.X, severity)
        self.NoisyX = self.NoisyX / 255.0

        if args.dataset == 'mnist':
            self.NoisyX = self.NoisyX - 0.1307
            self.NoisyX = self.NoisyX / 0.3081

        elif args.dataset == 'cifar10':
            self.NoisyX = self.NoisyX - mean
            self.NoisyX = self.NoisyX / std

        else:
            raise NotImplementedError

        self.NoisyX = torch.FloatTensor(self.NoisyX)

    def __len__(self):
        return len(self.NoisyX)

    def __getitem__(self, item):
        x = self.NoisyX[item]
        y = self.Y[item]

        return x, y


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    print(x * c)
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.06, .1, .12, .16, .2][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
