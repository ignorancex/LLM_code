#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import faradaydreams.measurement_operator
import faradaydreams.convex_solvers

np.random.seed(42)


def test_constrained():

    lambda2 = np.linspace(0, 6, 128)
    weights = np.ones((len(lambda2)))
    dphi = 0.24
    Nd = 256
    phi = faradaydreams.measurement_operator.create_phi_array(Nd, dphi)

    matrix_op = faradaydreams.measurement_operator.faraday_operator(
        lambda2=lambda2,
        phi=phi,
        weights=weights,
        lambda2_width=None,
        spectral_axis=-1,
        nufft=False)

    nu_matrix_op = faradaydreams.measurement_operator.power_method(
        matrix_op, 1e-4)

    x_true = np.zeros(phi.shape)
    x_true[120] = 1
    y = matrix_op.dir_op(x_true)
    sigma = 0.0001
    noise = np.random.normal(0, sigma, lambda2.shape)
    y += noise

    z, diag = faradaydreams.convex_solvers.solver(
        faradaydreams.convex_solvers.algorithm.l1_constrained,
        y,
        sigma,
        matrix_op,
        wav=["dirac"],
        levels=1,
        operator_norm=nu_matrix_op,
        beta=1e-3,
        options={
            'tol': 1e-5,
            'iter': 50000,
            'update_iter': 5000,
            'record_iters': False,
            "positivity": False,
            'real': False,
            'project_positive_lambda2': False
        },
        viewer=None,
        estimate=None,
        spectral_axis=-1)
    assert (np.all(np.abs(x_true - z) < 1e-2))
