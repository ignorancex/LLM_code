from .hamiltonian_cycle_dataset import HamiltonianCycleDataset
from .navigation_dataset import NavigationDataset
from .sokoban_dataset import SokobanDataset
from ..configs import SIMULATOR
from ..simulators import *

Dataset = None

if SIMULATOR == Navigation:
    Dataset = NavigationDataset
elif SIMULATOR == HamiltonianCycle:
    Dataset = HamiltonianCycleDataset
else:
    Dataset = SokobanDataset
