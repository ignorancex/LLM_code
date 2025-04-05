#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import faradaydreams.measurement_operator as measurement_operator
import faradaydreams.convex_solvers as solvers
import faradaydreams.utilities as utilities


def recover_1d_faraday_spectrum(
        Q,
        U,
        error_Q,
        error_U,
        freq,
        dphi,
        Nphi,
        sigma_factor=1,
        beta=1e-1,
        options={
            'tol': 1e-5,
            'iter': 50000,
            'update_iter': 1000,
            'record_iters': False,
            'project_positive_lambda2': False
        },
        channel_widths=None):
    # Process data for the solver
    y, weights, sigma = utilities.get_polarization_and_weights(P_q=Q,
                                                               P_u=U,
                                                               error_q=error_Q,
                                                               error_u=error_U)
    # Scale the uncertainty if asked for
    sigma = sigma_factor * sigma
    # Calcualte the lambda^2 values
    lambda2, dlambda2 = utilities.calculate_lambda2(freq=freq,
                                                    dfreq=channel_widths)
    phi = measurement_operator.create_phi_array(Nphi, dphi)
    m_op = measurement_operator.faraday_operator(lambda2,
                                                 phi,
                                                 weights,
                                                 lambda2_width=dlambda2,
                                                 nufft=False)
    # Estimate largest eigenvalue of the measurement linear operator
    nu = measurement_operator.power_method(m_op)
    # Solve for the Faraday spectrum
    solution, diag = solvers.solver(algo=solvers.algorithm.l1_constrained,
                                    measurements=weights * y,
                                    sigma=sigma,
                                    m_op=m_op,
                                    wav=['dirac'],
                                    levels=1,
                                    operator_norm=nu,
                                    beta=beta,
                                    options=options,
                                    viewer=None,
                                    estimate=m_op.adj_op(y * weights),
                                    spectral_axis=-1)
    # Get lambda2 coordinates for the model space
    model_lambda2 = measurement_operator.create_lambda2_array(Nphi, dphi)
    # Calculate the model in lambda2 space
    m_op_model = measurement_operator.faraday_operator(
        model_lambda2,
        phi,
        np.ones(model_lambda2.shape),
        lambda2_width=None,
        nufft=False)
    y_model = m_op_model.dir_op(solution)
    return {
        "solution": solution,
        "phi": phi,
        "y_solution": y_model,
        "solution_lambda2": model_lambda2,
        "measurements": y,
        "measurements_residuals": y - m_op.dir_op(solution) / weights,
        "measurements_lambda2": lambda2,
        "measurements_weights": weights / sigma,
        "measurement_operator": m_op
    }
