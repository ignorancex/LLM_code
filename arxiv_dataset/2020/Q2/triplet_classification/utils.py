#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import itertools
import numpy as np
import numpy.random as rand
from numpy.random import randint as rint
from sklearn import preprocessing as pp
from sklearn.model_selection import KFold
from pdb import set_trace as st


def printr(s):
    print('\r{}'.format(' ' * 75), end='')
    print('\r{}'.format(s), end='')


def load_dataset(dataset='toy'):
    Xs = np.concatenate((
        rand.multivariate_normal([1, 1], [[1, 0], [0, 1]], 1000),
        rand.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 1000)))
    Xtest = np.concatenate((
        rand.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100),
        rand.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 100)))
    ys = np.concatenate((np.ones(1000), -1 * np.ones(1000)))
    ytest = np.concatenate((np.ones(100), -1 * np.ones(100)))
    return Xs, ys, Xtest, ytest


def gen_knn_dataset(nt=1000, prior=0.7, dataset='toy'):
    Xs, ys, Xtest, ytest = load_dataset(dataset=dataset)
    Xs = Xs.reshape((len(Xs), -1))
    Xtest = Xtest.reshape((len(Xtest), -1))
    Xs_ids = np.array(range(len(Xs)))
    Xp_ids = Xs_ids[ys == 1]
    Xn_ids = Xs_ids[ys == -1]
    lXp = len(Xp_ids)
    lXn = len(Xn_ids)

    n1 = int(nt * (1 - prior * (1 - prior)))
    n2 = nt - n1
    n1a = int(nt * prior * prior * prior)
    n1b = int(nt * prior * prior * (1 - prior))
    n1c = int(nt * prior * (1 - prior) * (1 - prior))
    n1d = n1 - n1a - 2 * (n1b + n1c)
    n2a = int(nt * prior * prior * (1 - prior))
    n2b = n2 - n2a

    n1a1s = Xp_ids[rint(lXp, size=n1a)]
    n1a2s = Xp_ids[rint(lXp, size=n1a)]
    n1a3s = Xp_ids[rint(lXp, size=n1a)]

    n1b1s = Xp_ids[rint(lXp, size=n1b)]
    n1b2s = Xp_ids[rint(lXp, size=n1b)]
    n1b3s = Xn_ids[rint(lXn, size=n1b)]

    n1c1s = Xp_ids[rint(lXp, size=n1c)]
    n1c2s = Xn_ids[rint(lXn, size=n1c)]
    n1c3s = Xn_ids[rint(lXn, size=n1c)]

    n1d1s = Xn_ids[rint(lXn, size=n1b)]
    n1d2s = Xp_ids[rint(lXp, size=n1b)]
    n1d3s = Xp_ids[rint(lXp, size=n1b)]

    n1e1s = Xn_ids[rint(lXn, size=n1c)]
    n1e2s = Xn_ids[rint(lXn, size=n1c)]
    n1e3s = Xp_ids[rint(lXp, size=n1c)]

    n1f1s = Xn_ids[rint(lXn, size=n1d)]
    n1f2s = Xn_ids[rint(lXn, size=n1d)]
    n1f3s = Xn_ids[rint(lXn, size=n1d)]

    n2a1s = Xp_ids[rint(lXp, size=n2a)]
    n2a2s = Xp_ids[rint(lXp, size=n2a)]
    n2a3s = Xn_ids[rint(lXn, size=n2a)]

    n2b1s = Xn_ids[rint(lXn, size=n2b)]
    n2b2s = Xn_ids[rint(lXn, size=n2b)]
    n2b3s = Xp_ids[rint(lXp, size=n2b)]

    cons_a = np.concatenate((n1a1s, n1b1s, n1c1s, n1d1s,
                             n1e1s, n1f1s, n2a1s, n2b1s))
    cons_b = np.concatenate((n1a2s, n1b2s, n1c2s, n1d2s,
                             n1e2s, n1f2s, n2a2s, n2b2s))
    cons_c = np.concatenate((n1a3s, n1b3s, n1c3s, n1d3s,
                             n1e3s, n1f3s, n2a3s, n2b3s))
    cons = (cons_a, cons_b, cons_a, cons_c)

    return Xs, cons, Xtest, ytest


def gen_dataset(nt=1000, prior=0.7, dataset='toy'):
    Xs, ys, Xtest, ytest = load_dataset(dataset=dataset)
    Xp = Xs[ys == 1]
    Xn = Xs[ys == -1]
    lXp = len(Xp)
    lXn = len(Xn)

    n1 = int(nt * (1 - prior * (1 - prior)))
    n2 = nt - n1
    n1a = int(nt * prior * prior * prior)
    n1b = int(nt * prior * prior * (1 - prior))
    n1c = int(nt * prior * (1 - prior) * (1 - prior))
    n1d = n1 - n1a - 2 * (n1b + n1c)
    n2a = int(nt * prior * prior * (1 - prior))
    n2b = n2 - n2a

    Xt1 = np.concatenate(
        (np.hstack((Xp[rint(lXp, size=n1a)],
                    Xp[rint(lXp, size=n1a)],
                    Xp[rint(lXp, size=n1a)])),
         np.hstack((Xp[rint(lXp, size=n1b)],
                    Xp[rint(lXp, size=n1b)],
                    Xn[rint(lXn, size=n1b)])),
         np.hstack((Xp[rint(lXp, size=n1c)],
                    Xn[rint(lXn, size=n1c)],
                    Xn[rint(lXn, size=n1c)])),
         np.hstack((Xn[rint(lXn, size=n1b)],
                    Xp[rint(lXp, size=n1b)],
                    Xp[rint(lXp, size=n1b)])),
         np.hstack((Xn[rint(lXn, size=n1c)],
                    Xn[rint(lXn, size=n1c)],
                    Xp[rint(lXp, size=n1c)])),
         np.hstack((Xn[rint(lXn, size=n1d)],
                    Xn[rint(lXn, size=n1d)],
                    Xn[rint(lXn, size=n1d)]))))

    Xt2 = np.concatenate(
        (np.hstack((Xp[rint(lXp, size=n2a)],
                    Xn[rint(lXn, size=n2a)],
                    Xp[rint(lXp, size=n2a)])),
         np.hstack((Xn[rint(lXn, size=n2b)],
                    Xp[rint(lXp, size=n2b)],
                    Xn[rint(lXn, size=n2b)]))))

    return Xt1.astype(np.float32), Xt2.astype(np.float32), Xtest.astype(np.float32), ytest[:, None]
