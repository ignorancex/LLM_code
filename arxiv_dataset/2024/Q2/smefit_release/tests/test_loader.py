# -*- coding: utf-8 -*-
"""Test loader module"""
import pathlib

import numpy as np

from smefit.loader import load_datasets

commondata_path = pathlib.Path(__file__).parents[0] / "fake_data"


exp_test_1 = {"data": [1, 2]}
exp_test_2 = {"data": [3, 4, 5, 6]}

theory_test_1 = {
    "best_sm": [1.0, 2.0],
    "theory_cov": [[0.1, 0.2], [0.2, 0.4]],
    "LO": {
        "SM": [2.0, 2.0],
        "Op1": [0.1, 0.2],
        "Op3": [0.5, 0.6],
        "Op2": [0.3, 0.4],
        "Op1*Op1": [0.7, 0.8],
        "Op2*Op2": [0.9, 0.11],
        "Op2*Op1": [0.12, 0.13],
        "Op2*Op3": [0.14, 0.15],
    },
    "NLO": {
        "SM": [2.0, 2.0],
        "Op1": [0.1, 0.2],
        "Op3": [0.5, 0.6],
        "Op2": [0.3, 0.4],
        "Op1*Op1": [0.7, 0.8],
        "Op2*Op2": [0.9, 0.11],
        "Op2*Op1": [0.12, 0.13],
        "Op2*Op3": [0.14, 0.15],
    },
}

theory_test_2 = {
    "best_sm": [1.1, 2.3, 3.0, 4.0],
    "theory_cov": [
        [
            [1.2, 0.4, 0.3, 0.2],
            [0.4, 1.2, 0.3, 0.2],
            [0.3, 0.3, 1.2, 0.3],
            [0.2, 0.2, 0.3, 0.2],
        ]
    ],
    "LO": {
        "SM": [1.5, 1.5, 1.5, 1.5],
        "Op1": [1.1, 1.2, 1.3, 1.4],
        "Op4": [2.5, 1.6, 1.7, 1.8],
        "Op2": [1.3, 4.4, 4.5, 4.6],
        "Op1*Op1": [1.7, 1.8, 1.9, 2.0],
        "Op2*Op2": [1.9, 1.11, 2.1, 2.2],
        "Op2*Op1": [3.12, 2.13, 3.1, 3.2],
        "Op2*Op4": [9.14, 5.15, 5.16, 5.17],
    },
    "NLO": {
        "SM": [1.0, 1.0, 1.0, 1.0],
        "Op1": [2.1, 2.2, 2.3, 2.4],
        "Op4": [3.5, 3.6, 2.7, 2.8],
        "Op2": [4.3, 4.4, 5.5, 5.6],
        "Op1*Op1": [2.7, 1.8, 2.9, 2.10],
        "Op2*Op2": [9.9, 5.11, 6.16, 6.17],
        "Op2*Op1": [1.12, 2.13, 6.18, 6.19],
        "Op2*Op4": [7.14, 4.15, 4.17, 4.18],
    },
}


def test_load_datasets():
    operators_to_keep = np.array(["Op1", "Op2", "Op4"])

    for use_quad in [True, False]:
        for order in ["LO", "NLO"]:
            datasets = [
                {"name": "data_test1", "order": f"{order}"},
                {"name": "data_test2", "order": f"{order}"},
            ]
            loaded_tuple = load_datasets(
                commondata_path,
                datasets=datasets,
                operators_to_keep=operators_to_keep,
                use_quad=use_quad,
                use_theory_covmat=True,
                use_t0=False,
                use_multiplicative_prescription=False,
                theory_path=commondata_path,
                rot_to_fit_basis=None,
            )

            # construct expected lin corr tables
            lin_corr_1 = np.asarray(
                [
                    theory_test_1[order]["Op1"],
                    theory_test_1[order]["Op2"],
                    [0.0, 0.0],
                ]
            ).T

            lin_corr_2 = np.asarray(
                [
                    theory_test_2[order]["Op1"],
                    theory_test_2[order]["Op2"],
                    theory_test_2[order]["Op4"],
                ]
            ).T

            lin_corr = np.concatenate((lin_corr_1, lin_corr_2))

            # construct expected data and sm predictions
            exp_test = np.concatenate((exp_test_1["data"], exp_test_2["data"]))
            sm = np.concatenate((theory_test_1["best_sm"], theory_test_2["best_sm"]))

            np.testing.assert_equal(loaded_tuple.Commondata, exp_test)
            np.testing.assert_equal(loaded_tuple.SMTheory, sm)
            np.testing.assert_equal(loaded_tuple.OperatorsNames, operators_to_keep)
            np.testing.assert_equal(loaded_tuple.ExpNames, ["data_test1", "data_test2"])
            np.testing.assert_equal(loaded_tuple.NdataExp, [2, 4])

            np.testing.assert_equal(loaded_tuple.LinearCorrections, lin_corr)

            if use_quad:
                # construct expected quad corr tables
                quad_corr_1 = np.asarray(
                    [
                        [
                            theory_test_1[order]["Op1*Op1"],
                            theory_test_1[order]["Op2*Op1"],
                            np.zeros(2),
                        ],
                        [
                            np.zeros(2),
                            theory_test_1[order]["Op2*Op2"],
                            np.zeros(2),
                        ],
                        [np.zeros(2), np.zeros(2), np.zeros(2)],
                    ]
                )
                quad_corr_1 = np.transpose(quad_corr_1, axes=(2, 0, 1))

                quad_corr_2 = np.asarray(
                    [
                        [
                            theory_test_2[order]["Op1*Op1"],
                            theory_test_2[order]["Op2*Op1"],
                            np.zeros(4),
                        ],
                        [
                            np.zeros(4),
                            theory_test_2[order]["Op2*Op2"],
                            theory_test_2[order]["Op2*Op4"],
                        ],
                        [np.zeros(4), np.zeros(4), np.zeros(4)],
                    ]
                )
                quad_corr_2 = np.transpose(quad_corr_2, axes=(2, 0, 1))

                quad_corr = np.concatenate((quad_corr_1, quad_corr_2))
                np.testing.assert_equal(loaded_tuple.QuadraticCorrections, quad_corr)


def test_operator_correction_sorted():
    operators_to_keep = np.array(["Op1", "Op3", "Op2"])
    for order in ["LO", "NLO"]:
        loaded_tuple = load_datasets(
            commondata_path,
            datasets=[{"name": "data_test1", "order": f"{order}"}],
            operators_to_keep=operators_to_keep,
            use_quad=True,
            use_theory_covmat=True,
            use_t0=False,
            use_multiplicative_prescription=False,
            theory_path=commondata_path,
            rot_to_fit_basis=None,
        )

        lin_corr = [
            theory_test_1[order]["Op1"],
            theory_test_1[order]["Op2"],
            theory_test_1[order]["Op3"],
        ]

        np.testing.assert_equal(loaded_tuple.LinearCorrections.T, lin_corr)

        quad_corr = np.asarray(
            [
                [
                    theory_test_1[order]["Op1*Op1"],
                    theory_test_1[order]["Op2*Op1"],
                    np.zeros(2),
                ],
                [
                    np.zeros(2),
                    theory_test_1[order]["Op2*Op2"],
                    theory_test_1[order]["Op2*Op3"],
                ],
                [np.zeros(2), np.zeros(2), np.zeros(2)],
            ]
        )
        quad_corr = np.transpose(quad_corr, axes=(2, 0, 1))

        np.testing.assert_equal(loaded_tuple.QuadraticCorrections, quad_corr)
