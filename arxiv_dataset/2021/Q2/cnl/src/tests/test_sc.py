import numpy as np
import scipy as sp
#from numba import jit

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging
from tqdm import tqdm, trange

from spn import *
import spn_c2

import unittest

class TestSC(unittest.TestCase):
    def test_grad(self):
        dim = 50
        p_dim = 50
        diag_A = np.zeros(dim)

        def _grad_at_imaginary_axis(x ,scale,sigma):
            ### z_all = x + a + igamma
            ### z = x + igamma
            thres = 1e-9
            a=0
            #diag_A = a*np.ones(dim, dtype=np.complex128)
            z_all = x + a + 1j*scale
            g = (z_all - sp.sqrt(z_all**2 - 4*sigma**2))/(2*sigma**2)
            ps_g = 2./(sigma*sp.sqrt(z_all**2 - 4*sigma**2))  - (z_all - sp.sqrt(z_all**2 - 4*sigma**2))/sigma**3

            if x + a == 0:
                g0 = 1j*(scale - sp.sqrt( scale**2 + 4*sigma**2))/(2*sigma**2)
                ps_g0 = 2./(sigma*sp.sqrt(scale**2+4*sigma**2)) + (scale - sp.sqrt(scale**2+4*sigma**2))/sigma**3
                ps_g0 *= -1j
                assert ( abs(g -g0) < thres)
                assert ( abs(ps_g- ps_g0)<thres)

                #print (g, g0)
            ### partial derivation on sigma

            sc = SemiCircular(dim, p_dim,scale)
            sc.set_params(diag_A, sigma)

            sc2 = spn_c2.SemiCircular(dim, p_dim,scale)
            sc2.set_params(diag_A, sigma)

            z = x + 1j*scale
            G, omega,omega_sc = sc.cauchy_subordination(z*np.eye(2, dtype=np.complex128), init_omega=1j*np.eye(2,dtype=np.complex128), init_G_sc=-1j*np.eye(2,dtype=np.complex128))
            G2, omega2, omega_sc2 = sc2.cauchy_subordination(z*np.ones(2, dtype=np.complex128),  init_omega=1j*np.ones(2,dtype=np.complex128), init_G_sc=-1j*np.ones(2,dtype=np.complex128))
            #print (g, G[0][0], G2[0])

            assert( np.allclose(g, G[0][0]) )
            assert(np.allclose(g, G2[0]))


            ps_G = sc.grad_subordination(z, G, omega,omega_sc)
            ps_G2 = sc2.grad_subordination(z, G2, omega2,omega_sc2)

            ps_g1 = ps_G[-1][0][0]
            ps_g2 = ps_G2[-1][0]
            np.allclose(ps_g, ps_g1)
            np.allclose(ps_g, ps_g2)

            #print("raw:", ps_g)
            #print("sc:",ps_g1)
            #print("sc2:", ps_g2)

        scales = np.random.uniform(low = 1e-3, high=2, size = 10)
        for x in np.linspace(0,1,10):
                for scale in [1e-1,2e-1, 1 ]:
                    for sigma in [0.1, 0.2, 1]:
                        _grad_at_imaginary_axis(x, scale, sigma)
