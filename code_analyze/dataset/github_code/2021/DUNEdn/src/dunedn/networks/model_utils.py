"""
    This module contains utility functions for networks in general.
"""
import torch.nn as nn


class MyDataParallel(nn.DataParallel):
    """Data Parallel wrapper that allows calling model's attributes."""

    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MyDDP(nn.parallel.DistributedDataParallel):
    """Distributed Data Parallel wrapper that allows calling model's attributes."""

    def __getattr__(self, name):
        try:
            return super(MyDDP, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
