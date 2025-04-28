#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import ascii
import numpy as np

# local
from modules.query import DatabaseQuery
from modules import simulations


def main2():
    cat = ascii.read('data/L0100N1504/Ref/snapshot28/satellites.txt')
    bounds = {'Mstar': (1e10, 1e13)}


def main():
    cat = simulations.Catalog('data/L0100N1504/Ref/snapshot28/satellites.txt')
    print(cat.colnames)
    try:
        print(cat.ngal)
    except AttributeError:
        pass
    bounds = {'Mstar': (1e10, 1e13)}
    mask = np.ones(cat['GalaxyID'].size, dtype=bool)
    for key, val in bounds.items():
        mask *= (cat[key] >= val[0]) & (cat[key] <= val[1])
    newcat = cat[mask]
    print(mask.sum(), cat['GalaxyID'][mask].size, newcat['GalaxyID'].size)
    cat.cut(bounds, in_place=True)
    x = [i for i, j in enumerate(cat['GalaxyID'])]
    print('x =', max(x), len(x))
    print(cat.ngal, cat['GalaxyID'].size, len(cat['GalaxyID']))



main()



