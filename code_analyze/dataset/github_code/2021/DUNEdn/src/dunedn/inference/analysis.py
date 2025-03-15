"""
    This module contains the wrapper function for the ``dunedn analysis``
    command.

    Example
    -------

    Analysis help output:

    .. code-block:: text

        $ dunedn analysis --help
        usage: dunedn analysis [-h] [--input INPUT] [--target TARGET] runcard

        Load reconstructed and target events and compute accuracy metrics.

        positional arguments:
          runcard               yaml configcard path

        optional arguments:
          -h, --help            show this help message and exit
          --input INPUT, -i INPUT
                                path to the denoised event file
          --target TARGET, -t TARGET
                                path to the target event file
"""
import logging
import numpy as np
import torch
from pathlib import Path
from dunedn import PACKAGE
from dunedn.geometry.helpers import evt2planes
from dunedn.training.metrics import DN_METRICS, ROI_METRICS, MetricsList
from dunedn.utils.utils import load_runcard

logger = logging.getLogger(PACKAGE + ".analysis")


def add_arguments_analysis(parser):
    """
    Adds inference subparser arguments.

    Parameters
    ----------
        - parser: ArgumentParser, inference subparser object
    """
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="path to the denoised event file",
        metavar="INPUT",
        dest="input_path",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        help="path to the target event file",
        metavar="TARGET",
        dest="target_path",
    )
    parser.add_argument("--task", help="task type, available options: dn|roi")
    parser.set_defaults(func=analysis)


def analysis(args):
    """Wrapper analysis function.

    Parameters
    ----------
    args: NameSpace
        Parsed from command line or from code.
    """
    return analysis_main(
        args.input_path,
        args.target_path,
        args.task,
    )


def analysis_main(
    input_path: Path,
    target_path: Path,
    task: str = "dn",
):
    """Inference main function.

    Loads an input event from file, makes inference and saves the ouptut.
    Eventually returns the output array.

    Parameters
    ----------
    input_path: Path
        Path to the denoised event file.
    target_path: Path
        Path to the target event file.
    task: str
        Performed task. Available options: dn|roi

    Returns
    -------
    np.array
        Ouptut event of shape=(nb wires, nb tdc ticks).
    """
    input_evt = np.load(input_path)[:, 2:]
    iinput, cinput = evt2planes(input_evt)

    target_evt = np.load(target_path)[:, 2:]
    itarget, ctarget = evt2planes(target_evt)

    metrics_names = DN_METRICS if task == "dn" else ROI_METRICS
    metrics_list = MetricsList(metrics_names)

    to_torch = lambda x: torch.Tensor(x)

    ires = metrics_list.compute_metrics(to_torch(iinput), to_torch(itarget))
    cres = metrics_list.compute_metrics(to_torch(cinput), to_torch(ctarget))

    res = metrics_list.combine_collection_induction_results(ires, cres)

    metrics_list.print_metrics(logger, res)
