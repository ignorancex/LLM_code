import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import axes3d


def one_hot(x, levels):
    """
    Output
    ------
    One hot Encoding of the input.
    """

    batch_size, channel, H, W = x.size()
    x = x.unsqueeze_(4)
    x = torch.ceil(x * (levels - 1)).long()
    onehot = torch.zeros(batch_size, channel, H, W, levels).float().scatter_(4, x, 1)

    return onehot


def one_hot_to_thermometer(x, levels):
    """
    Convert One hot Encoding to Thermometer Encoding.
    """

    thermometer = torch.ones(size=x.shape) - torch.cumsum(x, dim=4)

    return thermometer


def Thermometer(x, levels):
    """
    Output
    ------
    Thermometer Encoding of the input.
    """

    onehot = one_hot(x, levels)

    thermometer = one_hot_to_thermometer(onehot, levels)

    return thermometer


def data_to_therm(x, levels):
    # Application of thermometer encoding
    encoding = Thermometer(x, levels)
    # These 3 lines below recode the 'levels' dimension into tensor channels
    encoding = encoding.permute(0, 2, 3, 1, 4)
    encoding = torch.flatten(encoding, start_dim=3)
    encoding = encoding.permute(0, 3, 1, 2)
    return encoding


def data_to_therm_numpy(x, levels):
    x = torch.tensor(x)
    # Application of thermometer encoding
    encoding = Thermometer(x, levels)
    # These 3 lines below recode the 'levels' dimension into tensor channels
    encoding = encoding.permute(0, 2, 3, 1, 4)
    encoding = torch.flatten(encoding, start_dim=3)
    encoding = encoding.permute(0, 3, 1, 2)
    return encoding.numpy()


if __name__ == "__main__":
    pass
