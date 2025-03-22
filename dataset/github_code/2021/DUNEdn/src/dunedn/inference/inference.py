"""
    This module contains the wrapper function for the ``dunedn inference``
    command.

    Example
    -------

    Inference help output:

    .. code-block:: text

        $ dunedn inference --help
        usage: dunedn inference [-h] [-i INPUT] [-o OUTPUT] -m MODEL [--model_path CKPT] [--onnx] [--onnx_export] runcard

        Load event and make inference with saved model.

        positional arguments:
          runcard            yaml configcard path

        optional arguments:
          -h, --help         show this help message and exit
          -i INPUT           path to the input event file
          -o OUTPUT          path to the output event file
          -m MODEL           model name. Valid options: (uscg|gcnn|cnn|id)
          --model_path CKPT  (optional) path to directory with saved model
          --onnx             wether to use ONNX exported model
          --onnx_export      wether to export models to ONNX
"""
import logging
from copy import deepcopy
import numpy as np
from pathlib import Path
from .hitreco import DnModel
from dunedn.configdn import PACKAGE
from dunedn.utils.utils import load_runcard, add_info_columns

THRESHOLD = 3.5  # the ADC threshold below which the output is put to zero
# TODO: move this into some dunedn config file

# instantiate logger
logger = logging.getLogger(PACKAGE + ".inference")


def add_arguments_inference(parser):
    """
    Adds inference subparser arguments.

    Parameters
    ----------
        - parser: ArgumentParser, inference subparser object
    """
    parser.add_argument(
        "-i",
        type=Path,
        help="path to the input event file",
        metavar="INPUT",
        dest="input_path",
    )
    parser.add_argument("--output", "-o", type=Path, help="the output folder")
    parser.add_argument(
        "-m",
        help="model name. Valid options: (uscg|gcnn|cnn|id)",
        required=True,
        metavar="MODEL",
        dest="modeltype",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="(optional) path to directory with saved model",
        default=None,
        dest="ckpt",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="wether to use ONNX exported model",
        dest="should_use_onnx",
    )
    parser.add_argument(
        "--onnx_export",
        action="store_true",
        help="wether to export models to ONNX",
        dest="should_export_to_onnx",
    )
    parser.set_defaults(func=inference)


def inference(args):
    """Wrapper inference function.

    Parameters
    ----------
    args: NameSpace
        Parsed from command line or from code.

    Returns
    -------
    np.array
        Output event of shape=(nb wires, nb tdc ticks)
    """
    setup = load_runcard(args.output / "cards/runcard.yaml")
    output_folder = args.output.joinpath(f"models/{args.modeltype}")
    # check if output folder has the right directory structure
    output_folder.is_dir()

    return inference_main(
        setup,
        args.input_path,
        output_folder,
        args.modeltype,
        args.ckpt,
        should_use_onnx=args.should_use_onnx,
        should_export_to_onnx=args.should_export_to_onnx,
    )


def inference_main(
    setup,
    input_path,
    output_folder,
    modeltype,
    ckpt,
    should_use_onnx=False,
    should_export_to_onnx=False,
):
    """Inference main function.

    Loads an input event from file, makes inference and saves the ouptut.
    Eventually returns the output array.

    Parameters
    ----------
    setup: dict
        Settings dictionary.
    input_path: Path
        Path to the input event file.
    output_folder: Path
        Path to the output folder.
    modeltype: str
        Model name. Available options: uscg|gcnn|cnn|id.
    ckpt: path
        Directory with saved model.
    should_use_onnx: bool
        Wether to use onnx format.
    """
    model = DnModel(setup, modeltype, ckpt, should_use_onnx=should_use_onnx)

    if should_export_to_onnx:
        model.onnx_export(ckpt)
        exit(-1)

    logger.info(f"Denoising event at {input_path}")
    evt = np.load(input_path)[:, 2:]
    evt_dn = model.predict(evt)

    # comment the following line to avoid thresholding
    evt_dn = thresholding_dn(evt_dn)

    name = (input_path.name).split("_")
    name.insert(-1, "dn")
    name = "_".join(name)
    fname = output_folder / name

    # add info columns
    evt_dn = add_info_columns(evt_dn)

    # save reco array
    np.save(fname, evt_dn)
    logger.info(f"Saved output event at {fname}")


def thresholding_dn(evt, t=THRESHOLD):
    """Apply a threhosld to the denoised waveforms to smooth results.

    Parameters
    ----------
    evt: np.array
        Event of shape=(nb wires, nb tdc ticks).
    t: float
        Threshold.

    Returns
    -------
    np.array
        Thresholded event of shape=(nb wires, nb tdc ticks).
    """
    mask = np.abs(evt) <= t
    # bind evt_dn variable to a copy to prevent in place substitution
    evt = deepcopy(evt)
    evt[mask] = 0
    return evt
