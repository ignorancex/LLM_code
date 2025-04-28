"""
This module contains the following functions:
- Sparse Quadric.
- Max-k-sum-squared.
- Non-Sparse Quadric.
"""

import numpy as np


class SparseQuadratic(object):
    """ An implementation of the sparse quadric function. """

    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.s = s
        self.dim = n
        self.rng = np.random.RandomState()

    def __call__(self, x):
        if not len(x) == self.dim:
            raise ValueError('Error! Dimension of input must be'+str(self.dim)) 
        f_no_noise = np.dot(x[0:self.s], x[0:self.s])
        return f_no_noise + self.noiseamp * self.rng.randn()


class MaxK(object):
    """ An implementation of the max-k-squared-sum function. """

    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.dim = n
        self.s = s
        self.rng = np.random.RandomState()

    def __call__(self, x):
        idx = np.argsort(np.abs(x))
        idx2 = idx[self.dim - self.s:self.dim]
        f_no_noise = np.dot(x[idx2], x[idx2]) / 2
        return f_no_noise + self.noiseamp * self.rng.randn()


class NonSparseQuadratic(object):
    """ An implementation of the sparse quadric function. """

    def __init__(self, n, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.dim = n
        self.rng = np.random.RandomState()

    def __call__(self, x):
        f_no_noise = np.dot(x, x)
        return f_no_noise + self.noiseamp * self.rng.randn()
