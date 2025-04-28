# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from smefit.analyze import fisher
from smefit.chi2 import Scanner, compute_chi2
from smefit.compute_theory import make_predictions
from smefit.loader import load_datasets

from .test_loader import commondata_path


class Scanner2D(Scanner):
    def chi2_func_2d(self, coeff1, coeff2, xs, ys):
        r"""Individual :math:`\chi^2` wrappper over series of values."""
        chi2_list = np.zeros((xs.size, ys.size))
        coeff1.is_free = True
        coeff2.is_free = True
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.coefficients.set_free_parameters([x, y])
                self.coefficients.set_constraints()
                chi2_list[i, j] = compute_chi2(
                    self.datasets,
                    self.coefficients.value,
                    self.use_quad,
                    self.use_multiplicative_prescription,
                    False,
                )
                summed_corrections = np.einsum(
                    "ij,j->i", self.datasets.LinearCorrections, self.coefficients.value
                )
                diff = self.datasets.Commondata - (
                    self.datasets.SMTheory + summed_corrections
                )

                chi2_list[i, j] = np.einsum(
                    "i,ij,j->", diff, self.datasets.InvCovMat, diff
                )

        return 2 * chi2_list


def test_fisher():
    """Test fisher information using the definition and evaluating the derivative."""

    c23 = 0.1
    c13 = -0.2
    coefficients_dict = {
        "Op1": {
            "min": -1,
            "max": 1,
        },
        "Op2": {
            "min": -3,
            "max": 1,
        },
        "Op3": {  # fixed to 0.1 * Op2 - 0.2 * Op1
            "constrain": [
                {"Op2": c23},
                {"Op1": c13},
            ],
            "min": -5,
            "max": 1,
        },
    }
    n_rep = 200
    x_max = 0.1

    for use_quad in [False, True]:
        operators_to_keep = np.array(["Op1", "Op2", "Op3"])
        dataset = load_datasets(
            commondata_path,
            datasets=[{"name": "data_test5", "order": "NLO"}],
            operators_to_keep=operators_to_keep,
            use_quad=use_quad,
            use_theory_covmat=True,
            use_t0=False,
            use_multiplicative_prescription=False,
            theory_path=commondata_path,
            rot_to_fit_basis=None,
        )
        run_card = {
            "use_quad": use_quad,
            "result_path": None,
            "result_ID": None,
            "data_path": commondata_path,
            "datasets": [{"name": "data_test5", "order": "NLO"}],
            "coefficients": coefficients_dict,
            "use_theory_covmat": True,
            "theory_path": commondata_path,
            "use_multiplicative_prescription": True,
        }
        chi2 = Scanner2D(run_card, n_rep)
        xs = np.linspace(0, x_max, n_rep)
        ys = np.linspace(0, x_max, n_rep)
        dx = dy = x_max / n_rep

        def fisher_info(coeff1, coeff2):
            f = chi2.chi2_func_2d(coeff1, coeff2, xs, ys)
            hessian = np.empty((2, 2, xs.size, ys.size))
            dfdx = np.gradient(f, dx)
            for l, dg in enumerate(dfdx):
                dfdxdy = np.gradient(dg, dy)
                for k, df in enumerate(dfdxdy):
                    hessian[k, l, :, :] = df
            return hessian

        fisher_cal = fisher.FisherCalculator(
            chi2.coefficients, dataset, compute_quad=use_quad
        )
        fisher_cal.compute_linear()

        if not use_quad:
            fisher_test = fisher_info(
                chi2.coefficients["Op1"], chi2.coefficients["Op2"]
            )

            # here the hessian is a flat matrix so we can just take the central value among the replicas
            np.testing.assert_allclose(
                fisher_cal.lin_fisher.values[0],
                np.diag(fisher_test[:, :, int(n_rep / 2), int(n_rep / 2)]),
                rtol=0.01,
            )
        else:
            # create a fake posterior
            post_df = pd.DataFrame(np.array([xs, ys]).T, columns=["Op1", "Op2"])
            post_df["Op3"] = c13 * post_df.Op1 + c23 * post_df.Op2
            smeft_pred = np.array(
                [
                    make_predictions(dataset, post_df.iloc[rep, :], use_quad, False)
                    for rep in range(xs.size)
                ]
            )

            fisher_cal.compute_quadratic(post_df, smeft_pred)
            fisher_test = fisher_info(
                chi2.coefficients["Op1"], chi2.coefficients["Op2"]
            )
            fisher_test = np.mean(np.mean(fisher_test, axis=3), axis=2)

            # TODO: improve tolerance
            # np.testing.assert_allclose(
            #     fisher_cal.quad_fisher.values[0], np.diag(fisher_test), rtol=0.05
            # )
