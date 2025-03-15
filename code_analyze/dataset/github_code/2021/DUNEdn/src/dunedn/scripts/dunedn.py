"""
    This script is the DUNEdn package entry point. Parses the subcommands from
    command line and calls the appropriate function to run.

    Example
    -------

    Main help output:

    .. code-block:: text
    
        $ dunedn --help
        usage: dunedn [-h] {preprocess,train,inference,analysis} ...

        dunedn

        positional arguments:
          {preprocess,train,inference,analysis}
            preprocess          preprocess dataset of protodune events
            train               train model loading settings from configcard
            inference           load event and make inference with saved model
            analysis            load reconstructed and target events and compute accuracy metrics

        optional arguments:
          -h, --help            show this help message and exit


"""
import argparse
from time import time as tm
from dunedn.preprocessing.preprocess import add_arguments_preprocessing
from dunedn.training.denoise_training import add_arguments_training
from dunedn.inference.inference import add_arguments_inference
from dunedn.inference.analysis import add_arguments_analysis


def main():
    """Defines the DUNEdn main entry point."""
    parser = argparse.ArgumentParser(description="dunedn")

    subparsers = parser.add_subparsers()

    # preprocess dataset before training
    p_msg = "Preprocess dataset of protoDUNE events: dumps planes and training crops."
    p_subparser = subparsers.add_parser(
        "preprocess", description=p_msg, help=p_msg.lower().split(":")[0]
    )
    add_arguments_preprocessing(p_subparser)

    # train
    t_msg = "Train model loading settings from configcard."
    t_subparser = subparsers.add_parser(
        "train", description=t_msg, help=t_msg.lower().strip(".")
    )
    add_arguments_training(t_subparser)

    # inference
    dn_msg = "Load event and make inference with saved model."
    dn_subparser = subparsers.add_parser(
        "inference",
        description=dn_msg,
        help=dn_msg.lower().strip("."),
    )
    add_arguments_inference(dn_subparser)

    # analysis
    ana_msg = "Load reconstructed and target events and compute accuracy metrics."
    dn_subparser = subparsers.add_parser(
        "analysis",
        description=ana_msg,
        help=ana_msg.lower().strip("."),
    )
    add_arguments_analysis(dn_subparser)

    args = parser.parse_args()

    start = tm()
    # execute parsed function
    args.func(args)
    print(f"Program done in {tm()-start} s")


# TODO: deal with the distributed training in a separated sub-package folder
# TODO: (enhancement) introduce the hpt (HyperParameterTuning) subcommand
