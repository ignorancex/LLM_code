#!/usr/bin/env python

from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from keras.models import save_model
from tensorflow.keras.utils import plot_model

from randomised_filtering.classifier.model import get_categorical_model

DESC = dedent(
    """
    Print the model architecture as a figure.

    NOTE: The tool 'netron' (pip-installable) can be used to visualize
        the Keras model dump (.h5) which can be created with the option
        '--path-to-model-output-file'.
    """
)
EPILOG = ""


def build_argparser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument("path_to_output_file", help="Path to output image file.")

    p.add_argument("--num-classes", type=int, default=3, help="Number of classes.")
    p.add_argument("--input-length", type=int, default=66, help="Input length.")
    p.add_argument(
        "--orientation",
        choices=("horizontal", "vertical"),
        default="vertical",
        help="Orientation of the figure.",
    )

    p.add_argument(
        "--path-to-model-output-file",
        required=False,
        help="Path to .h5 output file.",
    )

    return p


ORIENTATION_MAP = {"horizontal": "LR", "vertical": "TB"}


def main():

    args = vars(build_argparser().parse_args())

    # build model
    model = get_categorical_model(
        num_classes=args["num_classes"],
        input_shape=(args["input_length"], 1),
    )

    # print model to file
    plot_model(
        model,
        to_file=args["path_to_output_file"],
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir=ORIENTATION_MAP[args["orientation"]],
    )

    if args["path_to_model_output_file"]:
        save_model(model, args["path_to_model_output_file"], save_format="h5")


if __name__ == "__main__":
    main()
