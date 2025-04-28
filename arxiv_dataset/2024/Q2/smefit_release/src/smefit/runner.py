# -*- coding: utf-8 -*-
import itertools
import pathlib
import shutil
import subprocess
import sys
from shutil import copyfile

import yaml

from .analyze.pca import RotateToPca
from .chi2 import Scanner
from .log import logging
from .optimize.analytic import ALOptimizer
from .optimize.mc import MCOptimizer
from .optimize.ultranest import USOptimizer

try:
    from mpi4py import MPI

    run_parallel = True
except ModuleNotFoundError:
    run_parallel = False

_logger = logging.getLogger(__name__)


class Runner:
    """
    Container for all the possible |SMEFiT| run methods.

    Init the root path of the package where tables,
    results, plot config and reports are stored.

    Parameters
    ----------
        run_card : dict
            run card dictionary
        single_parameter_fits : bool
            True for single parameter fits
        runcard_file : pathlib.Path, None
            path to runcard if already present
    """

    def __init__(
        self, run_card, single_parameter_fits, pairwise_fits, runcard_file=None
    ):
        self.run_card = run_card
        self.runcard_file = runcard_file
        self.single_parameter_fits = single_parameter_fits
        self.pairwise_fits = pairwise_fits
        self.setup_result_folder()

    def setup_result_folder(self):
        """
        Create result folder and copy the runcard there
        """
        # Construct results folder
        result_ID = self.run_card["result_ID"]
        result_folder = pathlib.Path(self.run_card["result_path"])
        if self.run_card["replica"] is not None:
            res_folder_fit = (
                result_folder / result_ID / f"replica_{self.run_card['replica']}"
            )
        else:
            res_folder_fit = result_folder / result_ID

        if res_folder_fit.exists():
            _logger.warning(f"{res_folder_fit} already found, overwriting old results")
            shutil.rmtree(res_folder_fit)
        subprocess.call(f"mkdir -p {res_folder_fit}", shell=True)

        # Copy yaml runcard to results folder or dump it
        # in case no given file is passed
        runcard_copy = result_folder / result_ID / f"{result_ID}.yaml"
        if self.runcard_file is None:
            with open(runcard_copy, encoding="utf-8") as f:
                yaml.dump(self.run_card, f, default_flow_style=False)
        else:
            copyfile(
                self.runcard_file,
                runcard_copy,
            )

    @classmethod
    def from_file(cls, runcard_file, replica=None):
        """
        Create Runner from a runcard file

        Parameters
        ----------
        runcard_file: pathlib.Path, str
            path to runcard
        replica: int
            replica number. Optional used only for MC

        Returns
        -------
            runner: `smefit.runner.Runner`
                instance of class Runner
        """
        config = {}
        # load file
        with open(runcard_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["replica"] = replica
        # set result ID to runcard name by default
        if "result_ID" not in config:
            config["result_ID"] = runcard_file.stem

        single_parameter_fits = config.get("single_parameter_fits", False)
        pairwise_fits = config.get("pairwise_fits", False)

        return cls(
            config, single_parameter_fits, pairwise_fits, runcard_file.absolute()
        )

    def get_optimizer(self, optimizer):
        """Return the seleted optimizer."""

        if optimizer == "NS":
            return self.ultranest
        elif optimizer == "MC":
            return self.mc
        elif optimizer == "A":
            return self.analytic
        raise ValueError(f"{optimizer} is not available")

    def analytic(self, config):
        """Sample the analytic linear solution."""

        a_opt = ALOptimizer.from_dict(config)
        a_opt.run_sampling()

    def ultranest(self, config):
        """Run a fit with Ultra Nest."""

        # add external modules to paths
        if "external_chi2" in config:
            external_chi2 = config["external_chi2"]
            for _, module in external_chi2.items():
                module_path = module["path"]
                path = pathlib.Path(module_path)
                base_path, stem = path.parent, path.stem
                sys.path = [str(base_path)] + sys.path

        if run_parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                ns_opt = USOptimizer.from_dict(config)
            else:
                ns_opt = None
            ns_opt = comm.bcast(ns_opt, root=0)
        else:
            ns_opt = USOptimizer.from_dict(config)

        ns_opt.run_sampling()

    def mc(self, config):
        """Run a fit with |MC|."""

        mc_opt = MCOptimizer.from_dict(config)
        mc_opt.run_sampling()
        mc_opt.save()

    def rotate_to_pca(self):
        """Rotate to |PCA| basis."""

        _logger.info("Rotate input basis to PCA basis")
        pca_rot = RotateToPca.from_dict(self.run_card)
        pca_rot.compute()
        pca_rot.update_runcard()
        pca_rot.save()

    def global_analysis(self, optimizer):
        """Run a global fit using the selected optimizer.

        Parameters
        ----------
        optimizer: string
            optimizer to be used (NS, MC or A)
        """
        config = self.run_card
        opt = self.get_optimizer(optimizer)
        opt(config)

    def single_parameter_analysis(self, optimizer):
        """Run a seried of single parameter fits for all the operators specified in the runcard.

        Parameters
        ----------
        optimizer: string
            optimizer to be used (NS, MC or A)
        """

        config = self.run_card

        # loop on all the coefficients
        for coeff in config["coefficients"].keys():
            single_coeff_config = dict(config)
            single_coeff_config["coefficients"] = {}

            # skip contrained coeffs
            if "constrain" in config["coefficients"][coeff]:
                _logger.info("Skipping constrained coefficient %s", coeff)
                continue

            # We define the new coefficient config for the individual fit
            new_coeff_config = {}

            # seach for a relation: loop on all the coefficients
            for coeff2, vals in config["coefficients"].items():
                # skip free coefficients
                if "constrain" not in vals:
                    continue

                # if fixed value, crash
                if isinstance(vals["constrain"], (int, float)):
                    raise ValueError(
                        "Fixed value constrain do not make sense for single parameter fits."
                    )

                constrain = (
                    vals["constrain"]
                    if isinstance(vals["constrain"], list)
                    else [vals["constrain"]]
                )

                new_constrain = []
                # Now we redefine the constraints
                # We only keep constraints proportional to the current coeff,
                # if not we skip the contribution as it is not relevant
                # for the current individual fit
                for addend in constrain:
                    if coeff in addend:
                        new_constrain.append(addend)

                if new_constrain:
                    new_coeff_config[coeff2] = {
                        "constrain": new_constrain,
                        "min": vals["min"],
                        "max": vals["max"],
                    }

            # add fitted coefficient
            new_coeff_config[coeff] = config["coefficients"][coeff]
            single_coeff_config["coefficients"] = new_coeff_config

            opt = self.get_optimizer(optimizer)
            opt(single_coeff_config)

    def pairwise_analysis(self, optimizer):
        """Run a series of pairwise parameter fits for all the operators specified in the runcard.

        Parameters
        ----------
        optimizer: string
            optimizer to be used only NS is supported
        """
        if optimizer != "NS":
            raise ValueError("Paiwise analysis is implemented only for NS.")

        config = self.run_card
        for c1, c2 in itertools.combinations(config["coefficients"].keys(), 2):
            pairwise_coeff_config = dict(config)
            pairwise_coeff_config["coefficients"] = {}
            pairwise_coeff_config["coefficients"][c1] = config["coefficients"][c1]
            pairwise_coeff_config["coefficients"][c2] = config["coefficients"][c2]

            opt = self.get_optimizer(optimizer)
            opt(pairwise_coeff_config)

    def run_analysis(self, optimizer):
        """Run either the global analysis or a series of single parameter fits using the selected optimizer.

        Parameters
        ----------
        optimizer: string
            optimizer to be used (NS, MC or A)
        """

        if self.single_parameter_fits:
            self.single_parameter_analysis(optimizer)
        elif self.pairwise_fits:
            self.pairwise_analysis(optimizer)
        else:
            self.global_analysis(optimizer)

    def chi2_scan(self, n_replica, compute_bounds):
        r"""Run an individual :math:`\chi^2` scan.

        Parameters
        ----------
        n_replica: int
            number of replicas to use.
            If 0 only the :math:`\chi^2` experimental data
            will be computed.
        compute_bounds: bool
            if True compute and save the :math:`\chi^2` bounds.
        """
        scan = Scanner(self.run_card, n_replica)
        if compute_bounds:
            scan.compute_bounds()
        scan.compute_scan()
        scan.plot_scan()
