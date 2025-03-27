"""
    This module contains the wrapper function for the ``dunedn train`` command.

    Example
    -------

    Train help output:

    .. code-block:: text

        $ dunedn train --help

        usage: dunedn train [-h] [--model {cnn,gcnn,uscg}] [--output OUTPUT] [--force] [--interactive]

        Train model loading settings from configcard.

        optional arguments:
          -h, --help            show this help message and exit
          --model {cnn,gcnn,uscg}, -m {cnn,gcnn,uscg}
                                the model to train
          --output OUTPUT, -o OUTPUT
                                output folder
          --force               overwrite existing files if present
          --interactive, -i     triggers interactive mode
"""
import logging
from pathlib import Path
from dunedn import PACKAGE
from dunedn.utils.ask_edit_card import ask_edit_card
from dunedn.utils.utils import load_runcard, check_in_folder

logger = logging.getLogger(PACKAGE + ".training")


def add_arguments_training(parser):
    """
    Adds training subparser arguments.

    Parameters
    ----------
    parser: ArgumentParser
        Training subparser object.
    """
    valid_models = ["cnn", "gcnn", "uscg"]
    parser.add_argument(
        "--model", "-m", help="the model to train", choices=valid_models
    )
    parser.add_argument("--output", "-o", type=Path, help="output folder", default=None)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="triggers interactive mode"
    )
    parser.set_defaults(func=training)


def training(args):
    """Wrapper training function.

    Parameters
    ----------
    args: NameSpace
        Command line parsed arguments.

    Returns
    -------
    float
        Minimum loss over training.
    float
        Uncertainty over minimum loss.
    str
        Best checkpoint file name.
    """
    if args.interactive:
        ask_edit_card(logger, args.output)
    # load runcard and setup output folder structure
    setup = load_runcard(args.output / "cards/runcard.yaml")
    check_in_folder(setup["output"] / f"models/{args.model}", args.force)

    # launch main training function
    training_main(args.model, setup)


def training_main(modeltype: str, setup: dict):
    """Main training function.

    Parameters
    ----------
    modeltype: str
        The model to be trained. Available options: cnn | gcnn | uscg.
    setup: dict
        Settings dictionary.
    """
    from dunedn.networks.gcnn.training import gcnn_training
    from dunedn.networks.uscg.training import uscg_training

    if modeltype in ["cnn", "gcnn"]:
        logger.info(f"Training {modeltype} network")
        gcnn_training(modeltype, setup)
    elif modeltype == "uscg":
        logger.info("Training Convolutional Neural Network")
        uscg_training(setup)
    else:
        raise NotImplementedError(f"model not implemented, found: {modeltype}")
