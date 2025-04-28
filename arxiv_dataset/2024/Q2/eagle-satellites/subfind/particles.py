#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import frogress
import h5py
import numpy as np
import os
from astropy.table import Table
from glob import glob
from matplotlib import pyplot as plt

# local
import utils

# my code
from plottools import plotutils
plotutils.update_rcParams()


def main(simulation='RefL0100N1504', snapshot=26):
    z = utils.snapz(snapshot=snapshot)
    print('z =', z, type(z))
    snapdir = 'snapshot_{0:03d}_z{1:06.2f}*'.format(snapshot, z)
    snapdir = snapdir.replace('.', 'p')
    path = os.path.join('..', 'particles', simulation, snapdir)
    print(path)
    catalogs = sorted(glob(os.path.join(path, '*.hdf5')))
    print(catalogs[0])
    print('{0} catalogs available'.format(len(catalogs)))
    for i, cat in frogress.bar(enumerate(catalogs), steps=len(catalogs)):
        read_tbl(cat)
        if i == 3:
            break
    print('Done')
    return


def read_tbl(tbl):
    data = Table.read(tbl, format='hdf5')
    return
    try:
        data = Table.read(tbl)
        print('Table {0} read successfully'.format(tbl))
    except ValueError:
        print('Failed reading table {0}'.format(tbl))
    return

main()


