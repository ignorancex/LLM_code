import pytest
from numpy import linalg as LA
import numpy as np
import optimusprimal.linear_operators as linear_operators
import faradaydreams.measurement_operator

np.random.seed(42)

def test_faraday_operator():
    lambda2 = np.linspace(0, 6, 20)
    weights = np.ones((20))
    dphi = 0.24
    Nd = 16
    phi = faradaydreams.measurement_operator.create_phi_array(Nd, dphi)

    matrix_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=False)
    nufft_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=True)

    dir_inp = np.zeros((phi.shape[0]))
    dir_inp[8] = 1
    output1 = matrix_op.dir_op(dir_inp)
    output2 = nufft_op.dir_op(dir_inp)
    assert(np.allclose(output1, 1., 1e-3))
    assert(np.allclose(output2, 1., 1e-3))

    dir_inp = np.zeros((phi.shape[0]))
    dir_inp[9] = 1
    output1 = matrix_op.dir_op(dir_inp)
    output2 = nufft_op.dir_op(dir_inp)
    expected = np.exp(2j * lambda2 * phi[9])
    assert(np.allclose(output1, expected, 1e-3))
    assert(np.allclose(output2, expected, 1e-3))

    dir_inp = np.zeros((10, phi.shape[0]))
    dir_inp[:, 8] = 1
    output1 = matrix_op.dir_op(dir_inp)
    output2 = nufft_op.dir_op(dir_inp)
    assert(np.allclose(output1, 1., 1e-3))
    assert(np.allclose(output2, 1., 1e-3))
    
    adj_inp = np.ones((lambda2.shape[0]))
    
    output1 = matrix_op.adj_op(adj_inp)
    output2 = nufft_op.adj_op(adj_inp)
    expected = np.sum(np.exp(-2j * lambda2[:, None] * phi[None, :]),axis=0)
    norm = np.mean(output2/expected)
    assert(np.allclose(output1, expected, 1e-3))
    assert(np.allclose(output2, expected, 1e-3))

def test_power_method():
    lambda2 = np.linspace(0, 6, 20)
    weights = np.random.normal(1, 1, (20))
    dphi = 0.24
    Nd = 16
    phi = faradaydreams.measurement_operator.create_phi_array(Nd, dphi)

    matrix_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=False)
    nufft_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=True)

    nu_matrix_op = faradaydreams.measurement_operator.power_method(matrix_op, 1e-4)
    nu, sol = linear_operators.power_method(matrix_op, np.ones((matrix_op.cols)), 1e-4)
    assert(np.allclose(nu, nu_matrix_op, 1e-6))
    nu_nufft_op = faradaydreams.measurement_operator.power_method(nufft_op, 1e-4)
    nu, sol = linear_operators.power_method(nufft_op, np.ones((nufft_op.cols)), 1e-4)
    assert(np.allclose(nu, nu_nufft_op, 1e-6))
