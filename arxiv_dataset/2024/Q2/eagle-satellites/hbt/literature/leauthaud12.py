#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from hmf import MassFunction
from numpy import array, log10
from uncertainties import ufloat, unumpy

# my code
import stattools


def shmr(Mstar, zbin='1'):
    #return func(Mstar, *bestfit(zbin))
    # the function
    f = func(Mstar, *bestfit(zbin))
    # weights from hmf


def func(Mstar, logM1, logM0, beta, delta, gamma):
    M0 = 10**logM0
    return logM1 + beta*(log10(Mstar)-logM0) + \
        (Mstar/M0)**delta / (1 + (M0/Mstar)**gamma) - 0.5


def bestfit(zbin='1'):
    # logM1, logM0, beta, delta, gamma
    params = {
        '1': [ufloat(12.520, 0.037), ufloat(10.916, 0.020),
              ufloat(0.457, 0.009), ufloat(0.566, 0.086),
              ufloat(1.53, 0.18)],
        '2': [ufloat(12.725, 0.032), ufloat(11.038, 0.019),
              ufloat(0.466, 0.009), ufloat(0.61, 0.13),
              ufloat(1.95, 0.25)],
        '3': [ufloat(12.722, 0.027), ufloat(11.100, 0.018),
              ufloat(0.470, 0.008), ufloat(0.393, 0.088),
              ufloat(2.51, 0.25)]
        }
    return array(params[zbin])

