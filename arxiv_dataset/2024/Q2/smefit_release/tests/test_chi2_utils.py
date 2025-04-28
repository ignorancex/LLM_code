# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pandas as pd

from smefit.analyze import chi2_utils
from smefit.chi2 import compute_chi2
from smefit.compute_theory import make_predictions
from smefit.fit_manager import FitManager

here = pathlib.Path(__file__).parent


class mock_FitManager:
    # fake a posterior
    max_val = 10
    min_val = -10
    xs = np.random.rand(100, 3) * (max_val - min_val) + min_val
    fit = FitManager(here, "fake_results")
    post_df = pd.DataFrame(xs, columns=["Op1", "Op2", "Op4"])
    post_df["Op3"] = -0.1 * post_df.Op1 + 0.2 * post_df.Op2

    fit.config["use_quad"] = False
    fit.load_datasets()
    fit.results = {"samples": post_df}


class Test_FitManager(mock_FitManager):
    def test_predictions(self):
        # at linear level averaging on the coefficients before or after the sum is the same ...
        pr_test = make_predictions(
            self.fit.datasets,
            np.mean(self.post_df.values, axis=0),
            self.fit.config["use_quad"],
            False,
        )
        np.testing.assert_allclose(np.mean(self.fit.smeft_predictions, axis=0), pr_test)

    def test_predictions_rep(self):
        # now test replica by replica
        self.fit.config["use_quad"] = True
        self.fit.load_datasets()
        pr_tesr = np.zeros_like(self.fit.smeft_predictions)
        for rep in range(self.fit.results["samples"].shape[0]):
            pr_tesr[rep] = make_predictions(
                self.fit.datasets,
                self.fit.results["samples"].iloc[rep, :],
                self.fit.config["use_quad"],
                False,
            )
        np.testing.assert_allclose(pr_tesr, self.fit.smeft_predictions)


class Test_Chi2tableCalculator(mock_FitManager):
    chi2_cal = chi2_utils.Chi2tableCalculator(None)

    def test_chi2(self):
        self.fit.config["use_quad"] = False
        ch2_df, _ = self.chi2_cal.compute(self.fit.datasets, self.fit.smeft_predictions)

        chi2_mean_test = compute_chi2(
            self.fit.datasets,
            np.mean(self.post_df.values, axis=0),
            self.fit.config["use_quad"],
            False,
        )
        np.testing.assert_allclose(chi2_mean_test, ch2_df["chi2"])

    def test_chi2_rep(self):
        # now test replica by replica
        self.fit.config["use_quad"] = True
        self.fit.load_datasets()
        _, chi2_rep = self.chi2_cal.compute(
            self.fit.datasets, self.fit.smeft_predictions
        )
        chi2_rep_test = np.zeros(self.fit.results["samples"].shape[0])
        for rep in range(self.fit.results["samples"].shape[0]):
            chi2_rep_test[rep] = compute_chi2(
                self.fit.datasets,
                self.fit.results["samples"].iloc[rep, :],
                self.fit.config["use_quad"],
                False,
            )
        np.testing.assert_allclose(
            chi2_rep_test / self.fit.datasets.NdataExp.sum(), chi2_rep
        )
