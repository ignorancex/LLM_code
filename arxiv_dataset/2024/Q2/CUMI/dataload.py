import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn import preprocessing
# from sklearn.model_selection import StratifiedKFold


class import_data(Dataset):
    def __init__(self, data_path):
        self.data = sio.loadmat(data_path)
        self.views = []
        for i in range(len(self.data['X'][0])):
            view_data = preprocessing.scale(self.data['X'][0][i].astype(np.float32))
            self.views.append(view_data)
        self.labels  = np.squeeze(self.data['Y']-1)
    def __len__(self):
        return self.data['Y'].shape[0]
    def __getitem__(self, idx):
        views = [view[idx] for view in self.views]
        label = self.labels[idx]
        sample = {'views': views, 'label': label}
        return sample


