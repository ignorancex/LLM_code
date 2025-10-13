"""
Dataloader utilities.
"""

import os

import torch


# Dataloader utilities


def set_dtld_seed(dtld, seed):
    """
    Set dataloader seed.

    Update the seed of the dataloader after its initialization.
    If the dataset associated with the dataloader has an attribute
    :attr:`dtld.dataset.rng`, that represents a random number generator
    (*torch.Generator*), it will be updated with the same seed.

    Parameters
    ----------
    dtld : torch.utils.data.Dataloader
        Dataloader.
    seed : int
        Seed.
    """
    if dtld.generator is None:
        dtld.generator = torch.Generator()
    dtld.generator.manual_seed(seed)
    rng = getattr(dtld.dataset, "rng", None)
    if rng is not None and isinstance(rng, torch.Generator):
        rng.manual_seed(seed)


# Dataloader workers utilities


def fix_cpu_affinity(worker_id):  # pylint: disable=W0613
    """
    Fix CPU affinity of workers.
    """
    os.sched_setaffinity(0, range(os.cpu_count()))  # pylint: disable=E1101


def set_worker_seed(worker_id):
    """
    Set seeds of dataloader's workers.
    """
    fix_cpu_affinity(worker_id)
    worker_info = torch.utils.data.get_worker_info()
    rng = getattr(worker_info.dataset, "rng", None)
    if rng is not None and isinstance(rng, torch.Generator):
        rng.manual_seed(worker_info.seed)


# Dataloader pack


class _DataloaderPackIterator:
    def __init__(self, dtlds_iter, n_iter):
        self.dtlds_iter = dtlds_iter
        self.n_iter = n_iter
        self.i = 0
        self.null = ((None, None), None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.n_iter:
            raise StopIteration
        self.i += 1
        return (next(dtld, self.null) for dtld in self.dtlds_iter)


class DataloaderPack:
    """
    Pack dataloaders and return an iterator.

    This class helps to synchronize seeds and to deal with dataloaders of
    different lengths.
    """

    def __init__(self, dtlds, drop_last=False):
        self.dtlds = dtlds
        self.n_subjects = len(self.dtlds)
        lengths = tuple(len(dtld) for dtld in self.dtlds)
        if drop_last:
            self.n_iter = min(lengths)
            self.n_total = self.n_subjects * self.n_iter
        else:
            self.n_iter = max(lengths)
            self.n_total = sum(lengths)

    def __len__(self):
        return self.n_total

    def __iter__(self):
        dtlds_iter = tuple(iter(dtld) for dtld in self.dtlds)
        return _DataloaderPackIterator(dtlds_iter, self.n_iter)

    def set_dtld_seed(self, seed):
        for dtld in self.dtlds:
            set_dtld_seed(dtld, seed)
