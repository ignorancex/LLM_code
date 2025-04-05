from cw import CompoundWishart as CW
import numpy as np
import matplotlib.pyplot as plt
import unittest

class TestCW(unittest.TestCase):
    def test_cw(self):
        dim = 100
        p_dim = 200

        cw = CW(dim, p_dim, scale = 1e-1)
        cw.b = np.random.random(p_dim) ###uniform [0,1]
        #cw.plot_density()
        sample = np.random.random(16)
        cw.grad_loss(sample)
