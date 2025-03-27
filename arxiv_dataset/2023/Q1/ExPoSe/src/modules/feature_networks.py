import torch as t

from .commons import Conv, ResidualBlock, BGNN
from ..simulators import *


def get_feature_network_for_hamiltonian_cycle():
    return BGNN()


def get_feature_network_for_sokoban():
    return t.nn.Sequential(Conv(Sokoban.n_channels, 32, 3, 1, 1),
                           ResidualBlock(32),
                           ResidualBlock(32),
                           Conv(32, 64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           t.nn.AdaptiveMaxPool2d((1, 1)),
                           t.nn.Flatten())


def get_feature_network_for_navigation():
    return t.nn.Sequential(Conv(Navigation.n_channels, 32, 3, 1, 1),
                           ResidualBlock(32),
                           ResidualBlock(32),
                           Conv(32, 64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           ResidualBlock(64),
                           t.nn.AdaptiveMaxPool2d((1, 1)),
                           t.nn.Flatten())


def get_feature_network(simulator_class):
    if simulator_class == Sokoban:
        return get_feature_network_for_sokoban()
    elif simulator_class == Navigation:
        return get_feature_network_for_navigation()
    elif simulator_class == HamiltonianCycle:
        return get_feature_network_for_hamiltonian_cycle()
    else:
        print('Invalid class detected: ', simulator_class.__name__)
