#!/usr/bin/env python

import os
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter

from randomised_filtering.classifier.streamline_loader import get_indices_from_json
from randomised_filtering.evaluation import (
    get_output_folders,
    get_meta_streamline_index,
    evaluate_subsets,
)


DESC = "Evaluates vote statistics across multiple subset sizes."
EPILOG = ""


OUTPUT_FOLDER_NAME = "output"


# amount of subsets used across all experiment instances
#   (purely theoretical max votes received by a streamline).
MAX_SUBSETS = 455


def build_argparser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "--json_path",
        required=False,
        help="Path to json file containing indices that are relevant for the "
        "statistics. If not specified, evaluate for all streamlines.",
    )
    return p


def main():
    args = vars(build_argparser().parse_args())

    filepath = os.getcwd()

    # get all relevant folders from the path (name must begin with OUTPUT_FOLDER_NAME)
    folders = get_output_folders(filepath, OUTPUT_FOLDER_NAME)

    # get vote distributions for each streamline across all subset sizes in one variable
    meta_streamline_index = get_meta_streamline_index(folders)

    if args.get("json_path"):
        json_path = args["json_path"]
        print("Analyzing only indices from", json_path)
        ind = get_indices_from_json(json_path, dtype=np.int32)

        # select index subset
        meta_streamline_index = np.array(meta_streamline_index)[ind]

    # evaluate relevant streamlines and write vote distributions to file
    evaluate_subsets(meta_streamline_index, MAX_SUBSETS, "all")


if __name__ == "__main__":
    main()
