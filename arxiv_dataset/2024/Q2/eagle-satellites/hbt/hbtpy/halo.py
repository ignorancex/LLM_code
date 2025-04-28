from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields
import os
import pandas as pd
import six
import warnings

from HBTReader import HBTReader

from .simulation import Simulation


class Halos(object):

    def __init__(self, sim, isnap, as_dataframe=True, force_isnap=False):
        """

        if `force_isnap` is True, a ValueError will be raised if
        virial quantities do not exist for snapshot `isnap`. Otherwise,
        the nearest snapshot will be used (raises UserWarning).

        """
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        # alias
        self.available_snapshots = self.sim.virial_snapshots
        self.force_isnap = force_isnap
        self.isnap = self._set_isnap(isnap)
        self._catalog = None

    @property
    def catalog(self):
        if self._catalog is None:
            self._catalog = h5py.File(self.filename, 'r')['HostHalos']
        return self._catalog

    @property
    def filename(self):
        return os.path.join(self.sim.path, 'HaloSize',
                            'HaloSize_{0}.hdf5'.format(self.isnap))

    ### private methods

    def _set_isnap(self, isnap):
        if isnap < 0:
            isnap = self.sim.snapshots[isnap]
        if isnap not in self.available_snapshots:
            if self.force_isnap:
                msg = 'Halo sizes not present for snapshot {0}'.format(
                    isnap)
                raise ValueError(msg)
            j = np.argmin(abs(isnap-self.available_snapshots))
            isnap_used = self.available_snapshots[j]
            warn = 'Snapshot {0} does not have halo information. Using' \
            ' snapshot {1} instead.'.format(isnap, isnap_used)
            warnings.warn(warn)
            isnap = isnap_used
        return isnap
