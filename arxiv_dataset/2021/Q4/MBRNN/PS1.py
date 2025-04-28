import logging

import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    """
        Load dataset
    """

    def __init__(self, phase, opt, rseed=934):
        self.logger = logging.getLogger(__name__)

        self.rseed = rseed
        self.phase = phase
        self.ncls = opt.ncls
        self.bin_dn = opt.bin_dn
        self.data_dn = opt.data_dn
        self.train = opt.train

        self.X, self.y, self.z, self.zcls, self.binc = self._load_data()

        self.len = self.X.size()[0]

    def zbinning(self, z, bins=None):
        _z = z.ravel()

        fbin = '%s/galaxy_redshifts_%s-uniform.txt' % \
            (self.bin_dn, self.ncls)
        log_msg = "Read uniform bins for %s from %s" % (self.phase, fbin)
        self.bins = np.genfromtxt(fbin)

        self.logger.info(log_msg)

        self.bin_ids = np.digitize(_z, self.bins)-1
        self.bin_center = np.array([(lb+ub)/2. for lb, ub in
                                   zip(self.bins[:-1], self.bins[1:])])

        return self.bin_ids, self.bin_center

    def _zfilter(self, z):
        z[z < 0] = 0

        return z

    def _load_data(self):
        self.fn = '%s/%s.npy' % (self.data_dn, self.phase)
        self.logger.info("load data from %s" % self.fn)

        data = np.load(self.fn, allow_pickle=True).T
        types = data[0]

        X = np.array(data[3:].T, dtype=np.float32)
        y = np.array(types, dtype=np.long)
        z = np.array(data[1].reshape(-1, 1), dtype=np.float32)

        z = self._zfilter(z)

        zcls, binc = self.zbinning(z)

        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(z), \
            torch.from_numpy(zcls), torch.from_numpy(binc)

    def __getitem__(self, index):
        X, zcls = self.X[index], self.zcls[index]
        return X, zcls

    def __len__(self):
        return self.len
