# -*-encode: utf-8-*-
import unittest
from timer import *
import numpy as np
from cy_fde_spn import *
from spn_c2 import *
import logging

import env_logger


class TestCython(unittest.TestCase):
    def test_cy_test_all(self):
        result = 1
        result = cy_test_all(result)
        assert result

    def test_cy_cauchy_subordianation(self):
        timer = Timer()

        Z = np.asarray( [0.1j, 0.1j], dtype=np.complex)
        o_G = np.asarray( [-1j, -1j], dtype=np.complex)
        max_iter = 1000
        thres = 1e-8
        p_dim = 50
        dim = 50
        sigma = 0.1
        timer.tic()
        o_forward_iter = cy_cauchy_2by2( p_dim, dim,  sigma, Z, \
        max_iter,thres,\
        o_G)
        timer.toc()
        logging.info("forward_iter={}".format(o_forward_iter))
        logging.info("o_G={}".format(o_G))
        logging.info("{} sec".format(timer.total_time))
        #import pdb; pdb.set_trace()

        B = np.asarray( [0.1j, 0.1j], dtype=np.complex)
        o_omega = np.asarray( [0.1j, 0.1j], dtype=np.complex)
        o_omega_sc = np.asarray( [0.1j, 0.1j], dtype=np.complex)

        o_G_sc = np.asarray( [-1j, -1j], dtype=np.complex)
        max_iter = 1000
        thres = 1e-8
        p_dim = 50
        dim = 50
        sigma = 0.1
        o_forward_iter = 0
        a = 0.2*np.ones(dim)
        a = np.asarray(a, dtype=float)

        timer.tic()
        o_forward_iter = cy_cauchy_subordination(p_dim, dim,a,sigma,\
        B, \
        max_iter,thres,\
        o_G_sc,o_omega, o_omega_sc)
        timer.toc()
        logging.info("forward_iter={}".format(o_forward_iter))
        logging.info("o_G_sc={}".format(o_G_sc))
        logging.info("{} [sec]".format(timer.total_time))

    def test_cy_grad_cauchy_spn(self):
        timer = Timer()

        Z = np.asarray( [0.1j, 0.1j], dtype=np.complex)
        o_G = np.asarray( [-1j, -1j], dtype=np.complex)

        B = np.asarray( [0.1j, 0.1j], dtype=np.complex128)
        o_omega = np.asarray( [0.1j, 0.1j], dtype=np.complex128)
        o_omega_sc = np.asarray( [0.1j, 0.1j], dtype=np.complex128)

        o_G_sc = np.asarray( [-1j, -1j], dtype=np.complex128)
        max_iter = 1000
        thres = 1e-8
        p_dim = 50
        dim = 50
        sigma = 0.1
        a = 0.2*np.ones(dim)
        a = np.asarray(a, dtype=float)

        o_forward_iter = cy_cauchy_subordination(\
        p_dim, dim, a, sigma , \
        B,\
        max_iter,thres, \
        o_G_sc, o_omega, o_omega_sc)


        o_grad_sigma = np.zeros(2, dtype=np.complex128)
        o_grad_a = np.zeros(2*dim, dtype=np.complex128)

        z = Z[0]
        cy_timer = Timer()
        cy_timer.tic()
        cy_grad_cauchy_spn(p_dim ,dim, a, sigma,\
         z,o_G_sc, o_omega, o_omega_sc, o_grad_a, o_grad_sigma)
        cy_timer.toc()
        logging.info("c:Backward= {} sec".format(cy_timer.total_time))


        sc = SemiCircular(dim=dim, p_dim=p_dim)
        sc.set_params(a,sigma)
        sc.TEST_MODE = False
        py_timer = Timer()
        py_timer.tic()
        py_grad = sc.grad_subordination(z, o_G_sc, o_omega, o_omega_sc, CYTHON=False)
        py_timer.toc()
        logging.info("python:Backward= {} sec".format(py_timer.total_time))

        py_grad_a = py_grad[:dim, :].flatten()
        py_grad_sigma = py_grad[-1, :].flatten()

        assert( np.allclose(py_grad_a, o_grad_a))
        assert( np.allclose(py_grad_sigma, o_grad_sigma))


        sc.TEST_MODE = True
        py_grad = sc.grad_subordination(z, o_G_sc, o_omega, o_omega_sc, CYTHON=True)

    def test_cy_grad_cauchy_spn(self):

        sample = np.asarray([0.2], dtype=np.float64)
        num_sample = len(sample)
        p=50
        d=50
        scale = 1e-1
        a = np.random.uniform(0, 1, d)
        a = np.asarray(a, dtype=np.float64)
        sigma = 0.199820
        cy_grad_a = np.zeros(d, dtype=np.float64)
        cy_grad_sigma = np.zeros(1, dtype=np.float64)
        cy_loss = np.zeros(1, dtype=np.float64)



        B = np.ones(2,dtype=np.complex128)
        z = sample[0] + scale*1j
        w = sp.sqrt(z)
        logging.info("py w:{}".format( w))
        B *= w
        o_omega = np.asarray( [1j, 1j], dtype=np.complex)
        o_omega_sc = np.asarray( [1j, 1j], dtype=np.complex)
        o_G_sc = np.asarray( [-1j, -1j], dtype=np.complex)

        o_forward_iter = cy_cauchy_subordination(p, d,a,sigma,\
        B, \
        1000,1e-8,\
        o_G_sc,o_omega, o_omega_sc)

        logging.info("(test_grad_loss)o_G_sc:{}".format( o_G_sc))

        cy_forward_iter=\
        cy_grad_loss_cauchy_spn( p, d, a, sigma, scale,\
        num_sample, sample, \
        cy_grad_a, cy_grad_sigma, cy_loss)



        sc = SemiCircular(dim=d, p_dim=p)
        sc.set_params(a,sigma)
        sc.TEST_MODE = False
        sc.scale = scale


        grad, loss =sc.grad_loss_subordination( sample, False)
        #import pdb; pdb.set_trace()

        assert( np.allclose(grad[:d], cy_grad_a ))
        assert( np.allclose(grad[-1], cy_grad_sigma ))
