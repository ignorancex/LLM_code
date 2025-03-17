import pickle

import h5py
import numpy as np
from torch.utils.data import Dataset


class NavigationDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        if file.endswith('h5'):
            data = h5py.File(file, mode='r')
        else:
            f = open(file, 'rb')
            data = pickle.load(f, encoding='latin1')
            f.close()

        self.states = np.asarray(data['states'])
        self.actions = np.asarray(data['actions'])
        self.values = np.asarray(data['values' if 'values' in data else 'q_values'])

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.states[idx].astype(np.float32), self.actions[idx], self.values[idx]
