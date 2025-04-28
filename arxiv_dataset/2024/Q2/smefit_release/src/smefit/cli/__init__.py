# -*- coding: utf-8 -*-
import pathlib
import sys

import click

from .. import log
from ..analyze import run_report
from ..log import print_banner, setup_console
from ..postfit import Postfit
from ..prefit import Prefit
from ..projections import Projection
from ..runner import Runner
from .base import base_command

try:
    from mpi4py import MPI

    run_parallel = True
except ModuleNotFoundError:
    run_parallel = False

fit_card = click.argument(
    "fit_card",
    type=click.Path(path_type=pathlib.Path, exists=True),
)

n_replica = click.option(
    "-n",
    "--n_replica",
    type=int,
    default=None,
    required=True,
    help="Number of the replica",
)

log_file = click.option(
    "-l",
    "--log_file",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    required=False,
    help="path to log file",
)

rotate_to_pca = click.option(
    "--rotate_to_pca",
    type=bool,
    is_flag=True,
    default=False,
    help="Run the fit in the PCA basis",
)


@base_command.command("NS")
@fit_card
@log_file
@rotate_to_pca
def nested_sampling(
    fit_card: pathlib.Path, log_file: pathlib.Path, rotate_to_pca: bool
):
    """Run a fit with |NS| (Ultra Nest).

    Usage: smefit NS [OPTIONS] path_to_runcard
    """
    rank = 0
    if run_parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    if rank == 0:
        setup_console(log_file)
        print_banner()
        runner = Runner.from_file(fit_card.absolute())
        if rotate_to_pca:
            runner.rotate_to_pca()
        log.console.log("Running : Nested Sampling Fit ")
    else:
        runner = None

    if run_parallel:
        runner = comm.bcast(runner, root=0)
    runner.run_analysis("NS")


@base_command.command("A")
@fit_card
@log_file
@rotate_to_pca
def analytic_linear(
    fit_card: pathlib.Path, log_file: pathlib.Path, rotate_to_pca: bool
):
    """Get the analytic solution of the linear problem.

    Usage: smefit A [OPTIONS] path_to_runcard
    """
    setup_console(log_file)
    print_banner()
    runner = Runner.from_file(fit_card.absolute())
    if rotate_to_pca:
        runner.rotate_to_pca()
    log.console.log("Running : analytic solution.")
    runner.run_analysis("A")


@base_command.command("MC")
@fit_card
@n_replica
@log_file
@rotate_to_pca
def monte_carlo_fit(
    fit_card: pathlib.Path, n_replica: int, log_file: pathlib.Path, rotate_to_pca: bool
):
    """Run a fit with |MC|.

    Usage: smefit MC [OPTIONS] path_to_runcard
    """
    setup_console(log_file)
    print_banner()
    runner = Runner.from_file(fit_card.absolute(), n_replica)
    if rotate_to_pca:
        runner.rotate_to_pca()
    log.console.log("Running : MonteCarlo Fit")
    runner.run_analysis("MC")


@base_command.command("PREFIT")
@fit_card
def pre_fit(fit_card: pathlib.Path):
    """Run prefit: computes the SM chi2 as a check before fitting.

    Usage: smefit PREFIT [OPTIONS] path_to_runcard
    """
    runner = Runner.from_file(fit_card.absolute())
    prefit = Prefit(runner.run_card)
    prefit.chi2_sm()


@base_command.command("POSTFIT")
@click.argument(
    "result_folder",
    type=click.Path(path_type=pathlib.Path, exists=True),
)
@click.option(
    "-n",
    "--n_replica",
    type=int,
    default=None,
    required=True,
    help="total number of replica to save",
)
@click.option(
    "-c",
    "--clean_rep",
    is_flag=True,
    default=False,
    help="remove the replica file",
)
def post_fit(result_folder: pathlib.Path, n_replica: int, clean_rep: bool):
    """Run postfit selection over |MC| replicas.

    Usage: smefit POSTFIT [OPTIONS] path_to_result_folder
    """
    postfit = Postfit.from_file(result_folder.absolute())
    postfit.save(n_replica)
    if clean_rep:
        postfit.clean()


@base_command.command("SCAN")
@fit_card
@click.option(
    "-n",
    "--n_replica",
    type=int,
    default=0,
    required=False,
    help="number of replicas used during the scan, default (0) will use only experimental data",
)
@click.option(
    "-b",
    "--bounds",
    type=bool,
    is_flag=True,
    default=False,
    help="compute also the chi2 bounds",
)
def scan(fit_card: pathlib.Path, n_replica: int, bounds: bool):
    r"""Plot individual :math:`\chi^2` profiles for all the free parameters.

    Usage: smefit SCAN [OPTIONS] path_to_runcard
    """
    runner = Runner.from_file(fit_card.absolute())
    runner.chi2_scan(n_replica, bounds)


@base_command.command("R")
@click.argument(
    "report_card",
    type=click.Path(path_type=pathlib.Path, exists=True),
)
def report(report_card: pathlib.Path):
    """Run a fit report.

    Usage: smefit R path_to_runcard
    """
    run_report(report_card.absolute())


@base_command.command("PROJ")
@click.argument(
    "projection_card",
    type=click.Path(path_type=pathlib.Path, exists=True),
)
@click.option(
    "--lumi",
    type=float,
    default=None,
    required=False,
    help="Adjusts the statistical uncertainties according to the specified luminosity. "
    "If not specified, the original uncertainties are kept "
    "and the central values are fluctuates according to the specified noise level.",
)
@click.option(
    "--noise",
    type=click.Choice(["L0", "L1"], case_sensitive=True),
    default="L0",
    required=False,
    help="Noise level for the projection, choose between L0 or L1. Assumes L0 by default.",
)
def projection(projection_card: pathlib.Path, lumi: float, noise: str):
    r"""Compute projection for specified dataset"""

    projection_setup = Projection.from_config(projection_card)
    projection_setup.build_projection(lumi, noise)
