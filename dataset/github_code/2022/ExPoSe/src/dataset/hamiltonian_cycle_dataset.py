import numpy as np
import torch as t
from torch.utils.data import Dataset


class HamiltonianCycleDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        data = t.load(file)
        self.states = data['states']
        self.actions = data['actions']
        self.values = data['values']

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        edges, features = self.states[idx]
        return (edges.astype(np.float32), features.astype(np.float32)), self.actions[idx], self.values[idx]
