# -*- coding: utf-8 -*-
"""Test covmat module"""
import numpy as np
import pandas as pd

from smefit import covmat

stat = np.ones(3)
sys = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
covmat_test = np.diag(stat) + sys @ sys.T
sys_names = ["CORR", "CORR", "CORR", "CORR"]
sys_dataframe = pd.DataFrame(data=sys, columns=sys_names)


def test_construct_covmat():
    np.testing.assert_allclose(
        covmat.construct_covmat(stat, sys_dataframe), covmat_test
    )


stat1 = np.ones(3)
sys1 = np.array([[1, 0.5], [1, 0.5], [1, 0.5]])
sys1_names = ["CORR", "SPECIAL"]

sys_dataframe1 = pd.DataFrame(data=sys1, columns=sys1_names)

stat2 = np.ones(2)
sys2 = np.array([[0.5], [0.5]])
sys2_names = ["SPECIAL"]

sys_dataframe2 = pd.DataFrame(data=sys2, columns=sys2_names)

tot_cov = [
    [2.25, 1.25, 1.25, 0.25, 0.25],
    [1.25, 2.25, 1.25, 0.25, 0.25],
    [1.25, 1.25, 2.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 1.25, 0.25],
    [0.25, 0.25, 0.25, 0.25, 1.25],
]


def test_covmat_from_systematics():
    np.testing.assert_allclose(
        covmat.covmat_from_systematics(
            [stat1, stat2], [sys_dataframe1, sys_dataframe2]
        ),
        tot_cov,
    )
