# -*- coding: utf-8 -*-
"""Module for the computation of chi-squared values."""
import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from rich.progress import track

from . import compute_theory as pr
from .coefficients import CoefficientManager
from .loader import DataTuple, load_datasets
from .log import logging

_logger = logging.getLogger(__name__)


def compute_chi2(
    dataset,
    coefficients_values,
    use_quad,
    use_multiplicative_prescription,
    use_replica=False,
):
    r"""
    Compute the :math:`\chi^2`.

    Parameters
    ----------
    dataset : DataTuple
        dataset tuple
    coefficients_values : numpy.ndarray
        |EFT| coefficients values
    use_multiplicative_prescription: bool
        if True add the |EFT| contribution as a key factor
    use_quad: bool
        if True include also |HO| corrections
    rgemat: numpy.ndarray
        solution matrix of the RGE

    Returns
    -------
    chi2_total : float
        :math:`\chi^2` value
    """

    # compute theory prediction for each point in the dataset
    theory_predictions = pr.make_predictions(
        dataset, coefficients_values, use_quad, use_multiplicative_prescription
    )

    # compute experimental central values - theory
    if use_replica:
        diff = dataset.Replica - theory_predictions
    else:
        diff = dataset.Commondata - theory_predictions

    invcovmat = dataset.InvCovMat
    # note @ is slower when running with mpiexec
    return jnp.einsum("i,ij,j->", diff, invcovmat, diff)


class Scanner:
    r"""Class to compute and plot the idividual :math:`\chi^2` scan.

    Parameters
    ----------
    run_card : dict
        run card dictionary
    n_replica : int
        number of replica to use
    """

    def __init__(self, run_card, n_replica):

        self.n_replica = n_replica
        self.use_quad = run_card["use_quad"]
        self.result_path = f"{run_card['result_path']}/{run_card['result_ID']}"
        self.use_multiplicative_prescription = run_card.get(
            "use_multiplicative_prescription", False
        )
        self.datasets = load_datasets(
            run_card["data_path"],
            run_card["datasets"],
            run_card["coefficients"],
            run_card["use_quad"],
            run_card["use_theory_covmat"],
            False,
            self.use_multiplicative_prescription,
            run_card.get("default_order", "LO"),
            run_card.get("theory_path", None),
            run_card.get("rot_to_fit_basis", None),
            run_card.get("uv_couplings", False),
            run_card.get("external_chi2", False),
        )

        # set all the coefficients to 0
        self.coefficients = CoefficientManager.from_dict(run_card["coefficients"])
        self.coefficients.set_free_parameters(
            np.full(self.coefficients.free_parameters.shape[0], 0)
        )

        # build empty dict to store results
        self.chi2_dict = {}
        for name, row in self.coefficients.free_parameters.iterrows():
            self.chi2_dict[name] = {}
            self.chi2_dict[name]["x"] = np.linspace(row.minimum, row.maximum, 100)

    def regularized_chi2_func(self, coeff, xs, use_replica):
        r"""Individual :math:`\chi^2` wrappper over series of values.

        Parameters
        ----------
        coeff: `smefit.coefficient.Coefficient`
            coefficient to switch on.
        xs: numpy.array
            coeffient values.
        use_replica: bool
            if True compute the :math:`\chi^2` on |MC| replicas.

        Returns:
        --------
            individual reduced :math:`\chi^2` for each x value.

        """
        chi2_list = []
        coeff.is_free = True
        for x in xs:
            coeff.value = x
            self.coefficients.set_constraints()
            chi2_list.append(
                compute_chi2(
                    self.datasets,
                    self.coefficients.value,
                    self.use_quad,
                    self.use_multiplicative_prescription,
                    use_replica,
                )
            )
        return np.array(chi2_list) / self.datasets.Commondata.size

    def compute_bounds(self):
        r"""Compute individual bounds solving.

        ..math::
            \chi^2`- 2 = 0
        """

        # chi^2 - 3.841
        def chi2_func(xs):
            return self.regularized_chi2_func(coeff, xs, False) - 3.841

        # find the bound for each coefficient
        bounds = {}
        x0_interval = [-1000, 1000]
        for coeff in self.coefficients:
            if coeff.name not in self.chi2_dict:
                continue
            coeff.is_free = True
            roots = opt.newton(
                chi2_func,
                x0_interval,
                maxiter=400,
            )
            # test roots are not the same
            try:
                np.testing.assert_allclose(roots[0] - roots[1], 0, atol=1e-5)
                raise ValueError(
                    f"single bound found for {coeff.name}: {roots[0]} in range {x0_interval}."
                )
            except AssertionError:
                # test roots are sorted
                try:
                    np.testing.assert_allclose(roots, np.sort(roots))
                except AssertionError:
                    raise ValueError(
                        f"Bound found for {coeff.name}: {roots} are not sorted."
                    )

            # save bounds and update the x ranges
            bounds[coeff.name] = roots.tolist()
            self.chi2_dict[coeff.name]["x"] = np.linspace(roots[0], roots[1], 100)

            coeff.is_free = False
            coeff.value = 0.0
            _logger.info(f"chi^2 bounds for {coeff.name}: {roots}")

        with open(f"{self.result_path}/chi2_bounds.json", "w", encoding="utf-8") as f:
            json.dump(bounds, f)

    def compute_scan(self):
        r"""Compute the individual :math:`\chi^2` scan for each replica and coefficient."""
        # loop on replicas
        for rep in track(
            range(self.n_replica + 1),
            description="[green]Computing chi2 for each replica...",
        ):
            use_replica = rep != 0
            if use_replica:
                self.datasets = DataTuple(
                    self.datasets.Commondata,
                    self.datasets.SMTheory,
                    self.datasets.OperatorsNames,
                    self.datasets.LinearCorrections,
                    self.datasets.QuadraticCorrections,
                    self.datasets.ExpNames,
                    self.datasets.NdataExp,
                    self.datasets.InvCovMat,
                    np.random.multivariate_normal(
                        self.datasets.Commondata,
                        np.linalg.inv(self.datasets.InvCovMat),
                    ),
                )

            # loop on coefficients
            for coeff in self.coefficients:
                if coeff.name not in self.chi2_dict:
                    continue
                self.chi2_dict[coeff.name][rep] = self.regularized_chi2_func(
                    coeff, self.chi2_dict[coeff.name]["x"], use_replica
                )
                coeff.value = 0.0
                coeff.is_free = False

    def plot_scan(self):
        r"""Plot and save the :math:`\chi^2` scan for each coefficient."""
        # loop on coefficients
        for c, tab in self.chi2_dict.items():
            _logger.info(f"Plotting scan for {c}")
            plt.figure()
            for rep in range(self.n_replica + 1):
                chi2_min = np.array(tab[rep]).min()
                if rep == 0:
                    plt.plot(tab["x"], tab[rep] - chi2_min)
                else:
                    plt.plot(
                        tab["x"], tab[rep] - chi2_min, alpha=0.2, color="lightskyblue"
                    )

            plt.ylabel(r"$\chi^2 - \chi^2_{min}$")
            plt.hlines(
                0, tab["x"].min(), tab["x"].max(), ls="dotted", color="black", lw=0.5
            )
            plt.title(f"{c}")
            plt.tight_layout()
            plt.savefig(f"{self.result_path}/chi2_scan_{c}.png")
            plt.savefig(f"{self.result_path}/chi2_scan_{c}.pdf")
