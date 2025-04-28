# -*- coding: utf-8 -*-
"""Fitting the Wilson coefficients with |MC|."""
import time

import cma
import numpy as np
import scipy.optimize as opt
from rich.style import Style
from rich.table import Table

from .. import log
from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer

_logger = log.logging.getLogger(__name__)


class MCOptimizer(Optimizer):
    """Optimizer specification for |MC|.

    Parameters
    ----------
    loaded_datasets : `smefit.loader.DataTuple`
        dataset tuple
    coefficients : `smefit.coefficients.CoefficientManager`
        instance of `CoefficientManager` with all the relevant coefficients to fit
    result_path: pathlib.Path
        path to result folder
    use_quad : bool
        If True use also |HO| corrections
    result_ID : str
        result name
    single_parameter_fits : bool
        True for individual scan fits
    use_multiplicative_prescription : bool
        if True uses the multiplicative prescription for the |EFT| corrections
    replica : int
        replica number
    use_bounds : bool
        If true start the minimization with the specified values of min and max for each coeffient
    minimizer_specs : dict
        minimizer options. The allowed optrions are:

        Args:

        - mc_minimiser: minimizer alogrithm: 'cma', 'dual_annealing', 'trust-constr'.
        - maxiter: number of maximium iterations.
        - restarts: only for cma, number of restarts (< 9).
        - initial_temp: only for dual_annealing.
            The initial temperature, use higher values to facilitates a wider search
            of the energy landscape, allowing dual_annealing to escape local minima that it is trapped in.
            Default value is 5230. Range is (0.01, 5.e4].
        - restart_temp_ratio: only for dual_annealing.
            During the annealing process, temperature is decreasing,
            when it reaches initial_temp * restart_temp_ratio, the reannealing process is triggered.
            Default value of the ratio is 2e-5. Range is (0, 1).

        See also `cma.fmin`, `scipy.optimize.minimize`, `scipy.optimize.dual_annealing`.

    """

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        use_quad,
        result_ID,
        replica,
        single_parameter_fits,
        use_bounds,
        minimizer_specs,
        use_multiplicative_prescription,
        external_chi2=None,
    ):
        super().__init__(
            f"{result_path}/{result_ID}",
            loaded_datasets,
            coefficients,
            use_quad,
            single_parameter_fits,
            use_multiplicative_prescription,
            external_chi2,
        )
        self.chi2_values = []
        self.coeff_steps = []
        self.replica = replica
        self.epoch = 0
        self.use_bounds = use_bounds
        self.minimizer_specs = minimizer_specs

    @classmethod
    def from_dict(cls, config):
        """
        Create object from theory dictionary.
        The default minimizer is trust-constr.

        The minimizer options have to be specified with:

        ```
        mc_minimiser: 'cma'
        maxiter: 100000
        restarts: 0
        ```


        Parameters
        ----------
        config : dict
            configuration dictionary

        Returns
        -------
        Optimizer
            created object

        """

        loaded_datasets = load_datasets(
            config["data_path"],
            config["datasets"],
            config["coefficients"],
            config["use_quad"],
            config["use_theory_covmat"],
            config["use_t0"],
            config.get("use_multiplicative_prescription", False),
            config.get("default_order", "LO"),
            config.get("theory_path", None),
            config.get("rot_to_fit_basis", None),
            config.get("uv_couplings", False),
            config.get("external_chi2", False),
        )

        coefficients = CoefficientManager.from_dict(config["coefficients"])

        use_bounds = config.get("use_bounds", True)
        if not use_bounds:
            log.console.log("Running minimization without initial bounds")

        minimizer_specs = {}
        minimizer_specs["mc_minimiser"] = config.get("mc_minimiser", "trust-constr")

        if minimizer_specs["mc_minimiser"] == "cma":
            minimizer_specs["restarts"] = config.get("restarts", 0)
            if "restarts" not in config:
                _logger.warning("Using default no restarts")
        elif minimizer_specs["mc_minimiser"] == "dual_annealineg":
            minimizer_specs["restart_temp_ratio"] = config.get(
                "restart_temp_ratio", 2e-5
            )
            if "restart_temp_ratio" not in config:
                _logger.warning("Using default restert_temp_ratio: 2e-5")
            minimizer_specs["initial_temp"] = config.get("initial_temp", 5230)
            if "initial_temp" not in config:
                _logger.warning("Using default initial_temp: 5230")
        elif "mc_minimiser" not in config:
            _logger.warning("Using default minimizer 'trust-constr'")

        minimizer_specs["maxiter"] = config.get("maxiter", int(1e4))
        if "maxiter" not in config:
            _logger.warning("Setting maximum number of iterations (maxiter) to 1e4")

        single_parameter_fits = config.get("single_parameter_fits", False)
        use_multiplicative_prescription = config.get(
            "use_multiplicative_prescription", False
        )

        external_chi2 = config.get("external_chi2", None)

        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            config["replica"],
            single_parameter_fits,
            use_bounds,
            minimizer_specs,
            use_multiplicative_prescription,
            external_chi2,
        )

    def get_status(self, chi2):
        if len(self.chi2_values) == 0:
            self.chi2_values.append(chi2)

        if chi2 < self.chi2_values[-1]:
            self.chi2_values.append(chi2)
            self.coeff_steps.append(self.free_parameters.value)
            self.epoch += 1

    def chi2_func_mc(self, params, print_log=True):
        """
        Wrap the chi2 in a function for the optimizer. Pass noise and
        data info as args. Log the chi2 value and values of the coefficients.

        Parameters
        ----------
        params : np.ndarray
            noise and data info

        Returns
        -------
        np.ndarray
            chi2 value

        """
        self.coefficients.set_free_parameters(params)
        self.coefficients.set_constraints()
        current_chi2 = self.chi2_func(True, print_log)
        self.get_status(current_chi2)

        return current_chi2

    def run_sampling(self):
        """Run the minimization with |MC|."""

        t1 = time.time()

        maxiter = self.minimizer_specs["maxiter"]
        algorithm = self.minimizer_specs["mc_minimiser"]

        if algorithm == "cma":
            bounds = [None, None]
            if self.use_bounds:
                bounds = [self.free_parameters.minimum, self.free_parameters.maximum]
            cma.fmin(
                self.chi2_func_mc,
                self.free_parameters.value,
                sigma0=0.68,
                options={
                    "bounds": bounds,
                    "verbose": -1,
                    "verb_log": False,
                    "ftarget": self.npts,
                    "tolx": 1e-5,
                    "maxiter": maxiter,
                },
                bipop=self.minimizer_specs["restarts"] > 0,
                restarts=self.minimizer_specs["restarts"],
            )

        else:
            if self.use_bounds:
                bounds = opt.Bounds(
                    self.free_parameters.minimum, self.free_parameters.maximum
                )
            else:
                bounds = opt.Bounds(
                    np.full(self.free_parameters.shape[0], -np.inf),
                    np.full(self.free_parameters.shape[0], np.inf),
                )

            if algorithm == "dual_annealing":
                res = opt.dual_annealing(
                    self.chi2_func_mc,
                    bounds,
                    minimizer_kwargs={
                        "method": "trust-constr",
                        "bounds": bounds,
                        "options": {"maxiter": maxiter},
                    },
                    maxiter=maxiter,
                    x0=self.free_parameters.value,
                    initial_temp=self.minimizer_specs["initial_temp"],
                    restart_temp_ratio=self.minimizer_specs["restart_temp_ratio"],
                )
            else:
                res = opt.minimize(
                    self.chi2_func_mc,
                    self.free_parameters.value,
                    method="trust-constr",
                    bounds=bounds,
                    options={"maxiter": maxiter, "xtol": 1e-5},
                )
            _logger.info(res)
            if not res["success"]:
                raise ValueError("Minimization was not successful, exit ...")

        t2 = time.time()
        log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")

    def save(self):
        """
        Save MC replicas to json inside a dictionary:
        {coff: replica values}

        Parameters
        ----------
            result : dict
                result dictionary

        """
        self.coefficients.set_free_parameters(self.free_parameters.value)
        self.coefficients.set_constraints()
        values = {}
        if not self.single_parameter_fits:
            values["chi2"] = self.chi2_values[-1] / self.npts
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Parameter", style="bold red", no_wrap=True)
        table.add_column("Best value")
        for par, value in zip(self.coefficients.name, self.coefficients.value):
            table.add_row(f"{par}", f"{value:.3f}")
            values[par] = value
        log.console.print(table)

        posterior_file = (
            self.results_path
            / f"replica_{self.replica}/coefficients_rep_{self.replica}.json"
        )

        fit_result = {
            "samples": values,
            "logz": None,
            "max_loglikelihood": None,
            "best_fit_point": None,
        }

        self.dump_fit_result(posterior_file, fit_result)
