#!/usr/bin/env python

import os
import numpy as np
import cv2

from textwrap import dedent
from argparse import ArgumentParser, RawTextHelpFormatter

from randomised_filtering.classifier.model import (
    get_binary_model,
    get_categorical_model,
)
from randomised_filtering.classifier.streamline_loader import load_data
from randomised_filtering.classifier.training import training_cv

DESC = dedent(
    """
    Train a classifier with given tractogram as well as positive and negative votes.
"""
)

EPILOG = dedent(
    f"""
    Example call:
      {os.path.basename(__file__)} --tractogram data/599671/All_10M_corrected.trk \
        --positive data/599671/json/pos_streamlines.json \
        --negative data/599671/json/neg_streamlines.json \
        --p-vs-n --three-class
"""
)


DEFAULT_POINTS_PER_STREAMLINE = 23
DEFAULT_EPOCHS = 2
DEFAULT_FOLDS = 5
DEFAULT_BATCH_SIZE = 50


def build_argparser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )

    p.add_argument("--tractogram", required=True, help="Tractogram file (.trk)")
    p.add_argument(
        "--positive",
        required=True,
        help="Path to json file with indices of positive votes",
    )
    p.add_argument(
        "--negative",
        required=True,
        help="Path to json file with indices of negative votes",
    )
    p.add_argument(
        "--points-per-streamline",
        type=int,
        default=DEFAULT_POINTS_PER_STREAMLINE,
        help="Resample streamlines with given number of points (default: {}).".format(
            DEFAULT_POINTS_PER_STREAMLINE
        ),
    )
    p.add_argument(
        "--p-vs-n", action="store_true", help="Run binary experiment (P vs N)."
    )
    p.add_argument(
        "--p-vs-i", action="store_true", help="Run binary experiment (P vs I)."
    )
    p.add_argument(
        "--n-vs-i", action="store_true", help="Run binary experiment (N vs I)."
    )
    p.add_argument("--three-class", action="store_true", help="Run 3-class experiment.")
    p.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--folds",
        default=DEFAULT_FOLDS,
        type=int,
        help=f"Number of folds (default: {DEFAULT_FOLDS}).",
    )
    p.add_argument(
        "--epochs",
        default=DEFAULT_EPOCHS,
        type=int,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS}).",
    )
    return p


def print_args(args):
    print()
    print("Received command line arguments:")
    for k, v in args.items():
        print(f" - {k}: {v}")
    print()


def main():
    args = vars(build_argparser().parse_args())
    print_args(args)

    pos_streamlines, neg_streamlines, inc_streamlines = load_data(
        args["tractogram"],
        args["positive"],
        args["negative"],
        normalize=True,
    )

    np.random.shuffle(pos_streamlines)
    np.random.shuffle(neg_streamlines)
    np.random.shuffle(inc_streamlines)

    # resize data to fit network input dimensions
    resampling_shape = (3, args["points_per_streamline"])
    input_shape = (np.prod(resampling_shape), 1)  # shape of network input

    pos_resized = np.array(
        [cv2.resize(x, resampling_shape).reshape(input_shape) for x in pos_streamlines]
    )
    neg_resized = np.array(
        [cv2.resize(x, resampling_shape).reshape(input_shape) for x in neg_streamlines]
    )
    inc_resized = np.array(
        [cv2.resize(x, resampling_shape).reshape(input_shape) for x in inc_streamlines]
    )

    common_args = dict(
        input_shape=input_shape,
        batch_size=args["batch_size"],
        epochs=args["epochs"],
        nb_folds=args["nb_folds"],
    )

    if args["p_vs_n"]:
        run_plausible_vs_implausible(
            pos_resized, neg_resized, suffix="pn", **common_args
        )
    if args["p_vs_i"]:
        run_plausible_vs_implausible(
            pos_resized, neg_resized, suffix="pi", **common_args
        )
    if args["n_vs_i"]:
        run_plausible_vs_implausible(
            pos_resized, neg_resized, suffix="ni", **common_args
        )

    if args["three_class"]:
        run_multi_class(pos_resized, neg_resized, inc_resized, **common_args)


def run_plausible_vs_implausible(
    pos_resized,
    neg_resized,
    input_shape,
    batch_size=50,
    nb_folds=5,
    epochs=5,
    suffix="",
):

    # exchange one of these for inc_resized when wanting to train
    #   with inconclusive streamlines
    data = [neg_resized, pos_resized]
    model = get_binary_model(input_shape=input_shape)

    # train 5 models in 5-fold cross-validation
    training_cv(
        data=data,
        model=model,
        nb_folds=nb_folds,
        batch_size=batch_size,
        epochs=epochs,
        base_path_to_model=f"model_binary{suffix}",
    )


def run_multi_class(
    pos_resized,
    neg_resized,
    inc_resized,
    input_shape,
    batch_size=60,
    nb_folds=5,
    epochs=2,
):

    data = [neg_resized, pos_resized, inc_resized]
    model = get_categorical_model(num_classes=len(data), input_shape=input_shape)

    # train 5 models in 5-fold cross-validation
    training_cv(
        data=data,
        model=model,
        nb_folds=nb_folds,
        batch_size=batch_size,
        epochs=epochs,
        base_path_to_model="model_cat" + str(len(data)),
    )


if __name__ == "__main__":
    main()
