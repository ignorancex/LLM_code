import torch as t

from ..simulators import *


def get_value_network_for_hamiltonian_cycle():
    return t.nn.Sequential(t.nn.Conv2d(64, 1, 1),
                           t.nn.AdaptiveMaxPool2d((1, 1)),
                           t.nn.Flatten())


def get_value_network_for_sokoban():
    return t.nn.Sequential(t.nn.Linear(64, 1))


def get_value_network_for_navigation():
    return t.nn.Sequential(t.nn.Linear(64, 1))


def get_value_network(simulator_class):
    if simulator_class == Sokoban:
        return get_value_network_for_sokoban()
    elif simulator_class == Navigation:
        return get_value_network_for_navigation()
    elif simulator_class == HamiltonianCycle:
        return get_value_network_for_hamiltonian_cycle()
    else:
        print('Invalid class detected: ', simulator_class.__name__)
