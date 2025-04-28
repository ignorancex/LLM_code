import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MBRNN(nn.Module):
    '''
        Fully Connected Networks
    '''

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, opt):
        super(MBRNN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ncls_ = opt.ncls
        self.batch_size = opt.batch_size
        self.classes_ = np.arange(self.ncls_)

        ninp = opt.ninp
        ncls = opt.ncls

        broadening = list(map(int, opt.broadening.split(',')))
        narrowing = list(map(int, opt.narrowing.split(',')))

        broadening = [int(ninp)]+broadening
        arr = np.arange(len(broadening)-1)
        itr = np.column_stack((arr, arr, arr)).flatten()

        shared = [
            nn.Linear(broadening[j], broadening[j+1], bias=True)
            if i % 3 == 0 else nn.Softplus()
            if i % 3 == 1 else nn.BatchNorm1d(broadening[j+1])
            for i, j in enumerate(itr)]

        self.shared = nn.Sequential(*shared)
        self.shared.apply(self._init_weights)
        self.logger.info(self.shared)

        narrowing = [broadening[-1]]+narrowing+[ncls]
        carr = np.arange(len(narrowing)-1)
        citr = np.column_stack((carr, carr, carr)).flatten()

        zclassification = [
            nn.Linear(narrowing[j], narrowing[j+1], bias=True)
            if i % 3 == 0 else nn.Softplus()
            if i % 3 == 1 else nn.BatchNorm1d(narrowing[j+1])
            for i, j in enumerate(citr)]
        zclassification = zclassification[:-2]

        self.zclassification = nn.Sequential(*zclassification)
        self.zclassification.apply(self._init_weights)
        self.logger.info(self.zclassification)

    def forward(self, x):
        _x = self.shared(x)
        output = self.zclassification(_x)
        output = F.softmax(output, dim=1)
        return output
