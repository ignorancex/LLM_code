import os

from .simulators import *

environment = os.environ.get('SIMULATOR', None)
if environment == 'sokoban':
    SIMULATOR = Sokoban
elif environment == 'navigation':
    SIMULATOR = Navigation
elif environment == 'hamiltonian_cycle':
    SIMULATOR = HamiltonianCycle
else:
    raise Exception('Simulator not set.')
