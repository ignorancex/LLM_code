# -*- coding: utf-8 -*-
"""Test compute_theory and chi2 module"""
import numpy as np

from smefit import chi2, compute_theory, loader


def test_flatten():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    flatten_matrix = np.array([1, 2, 3, 5, 6, 9])
    np.testing.assert_allclose(compute_theory.flatten(matrix), flatten_matrix)


class TestPredictions:
    exp_data = np.array(
        [
            1.3,
            2.6,
        ]
    )
    sm_theory = np.array(
        [
            1,
            1.2,
        ]
    )
    operators_names = np.array(["Op1", "Op2"])
    lin_corr_values = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )  # Op1, Op2
    quad_corr_values = np.array(
        [
            [[0.3, 0.4], [0.0, 0.5]],
            [[0.6, 0.7], [0.0, 0.8]],
        ]
    )  # Op1^2, Op1*Op2, Op2^2
    covmat = np.array([[0.06, 0.02], [0.02, 0.03]])
    theory_covmat = np.array([[0.003, 0.001], [0.001, 0.002]])
    luminosity = 300
    replica = np.random.multivariate_normal(exp_data, covmat)

    dataset = loader.DataTuple(
        exp_data,
        sm_theory,
        operators_names,
        lin_corr_values,
        quad_corr_values,
        np.array(["exp1"]),
        np.array([exp_data.size]),
        np.linalg.inv(covmat),
        theory_covmat,
        luminosity,
        replica,
    )
    wilson_coeff = np.array([0.5, 0.6])

    def test_make_predictions(self):
        lin_corr = [
            # data0
            self.sm_theory[0]
            + self.wilson_coeff[0] * self.lin_corr_values[0, 0]
            + self.wilson_coeff[1] * self.lin_corr_values[0, 1],
            # data1
            self.sm_theory[1]
            + self.wilson_coeff[0] * self.lin_corr_values[1, 0]
            + self.wilson_coeff[1] * self.lin_corr_values[1, 1],
        ]
        np.testing.assert_allclose(
            compute_theory.make_predictions(
                self.dataset, self.wilson_coeff, False, False
            ),
            lin_corr,
        )
        quad_corr = np.array(
            [
                # data0
                +self.wilson_coeff[0] ** 2 * self.quad_corr_values[0, 0, 0]
                + self.wilson_coeff[0]
                * self.wilson_coeff[1]
                * self.quad_corr_values[0, 0, 1]
                + self.wilson_coeff[1] ** 2 * self.quad_corr_values[0, 1, 1],
                # data1
                +self.wilson_coeff[0] ** 2 * self.quad_corr_values[1, 0, 0]
                + self.wilson_coeff[0]
                * self.wilson_coeff[1]
                * self.quad_corr_values[1, 0, 1]
                + self.wilson_coeff[1] ** 2 * self.quad_corr_values[1, 1, 1],
            ]
        )
        quad_corr += lin_corr
        np.testing.assert_allclose(
            compute_theory.make_predictions(
                self.dataset, self.wilson_coeff, True, False
            ),
            quad_corr,
        )

    def test_compute_chi2(self):
        diff = self.exp_data - compute_theory.make_predictions(
            self.dataset, self.wilson_coeff, True, False
        )
        np.testing.assert_allclose(
            chi2.compute_chi2(self.dataset, self.wilson_coeff, True, False),
            diff @ np.linalg.inv(self.covmat) @ diff,
        )
