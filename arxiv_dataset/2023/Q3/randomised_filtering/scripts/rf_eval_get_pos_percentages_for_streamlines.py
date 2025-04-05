#!/usr/bin/env python

import os
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter

from randomised_filtering.classifier.streamline_loader import get_indices_from_json
from randomised_filtering.evaluation import (
    get_meta_streamline_index,
    get_output_folders,
    get_acceptance_rate
)


DESC = """
Retrieves percentage of positives votes received by streamlines across all ex-
periment instances/subset sizes. Either for all streamlines or a set of stream-
lines of interest.
"""
EPILOG = ""


OUTPUT_FOLDER_NAME = "output"

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

    # get relevant streamlines
    entries = meta_streamline_index
    name = ""
    if args.get("json_path"):
        jsonpath = args["json_path"]
        print("Analyzing only indices from", jsonpath)
        ind = get_indices_from_json(jsonpath, dtype=np.int32)
        entries = meta_streamline_index[ind]
        name = jsonpath.split(".")[0]

    # get acceptance rates for all streamlines of interests and write to file.
    result = np.array(
        [[entry[2], get_acceptance_rate(entry)] for entry in entries] 
    )  
    np.savetxt(f"percentages_{name}.txt", result)


if __name__ == "__main__":
    main()
