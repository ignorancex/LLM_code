import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MoleculeSet(Dataset):
    def __init__(self, dbpath, chpath, n_conformer, subset=None):
        self.dbpath = dbpath
        self.chpath = chpath
        self.n_conformer = n_conformer
        self.npzs = glob.glob("%s/*.%s" % (self.dbpath, "npz"))
        self.subset = subset
        os.makedirs(self.chpath, exist_ok=True)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = (
            idx if self.subset is None or len(idx) == 0 else np.array(self.subset)[idx]
        )
        return type(self)(
            dbpath=self.dbpath,
            chpath=self.chpath,
            n_conformer=self.n_conformer,
            subset=subidx,
        )

    def __len__(self):
        if self.subset is None:
            return len(self.npzs)
        return len(self.subset)

    def __getitem__(self, idx):
        properties = self.get_properties(idx)
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))
        return properties

    def _subset_id(self, idx):
        # print (idx)
        # get row
        if self.subset is None:
            npz_file = self.npzs[idx]
        else:
            npz_file = self.npzs[self.subset[idx]]
        return npz_file

    def get_properties(self, idx):
        properties = {}
        npz_file = self._subset_id(idx)
        # print (npz_file)
        with np.load(npz_file) as load:
            for key, value in load.items():
                if load[key].dtype == "float32":
                    properties[key] = torch.FloatTensor(value)
                else:
                    properties[key] = torch.LongTensor(value)
        return properties

    def clean_npzs(self):
        for npz_file in self.npzs:
            if os.path.exists(npz_file):
                os.remove(npz_file)
