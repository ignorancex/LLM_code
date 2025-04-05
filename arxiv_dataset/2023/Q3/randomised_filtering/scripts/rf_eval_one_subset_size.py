#!/usr/bin/env python

import os
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter

from randomised_filtering.classifier.streamline_loader import get_indices_from_json
from randomised_filtering.evaluation import process_subsets, evaluate_subsets


DESC = """
This script evaluates streamline votes in a folder containing both plausible/
implausible index .jsons and write main statistics them to a result file.
"""
EPILOG = ""


def build_argparser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "--folder_path",
        required=False,
        help="Path to experiment folder, as seen from cwd. (default: using `cwd`)",
    )
    p.add_argument(
        "--json_path",
        required=False,
        help="Path to json file containing indices that are relevant for "
        "the statistics. (default: evaluate for all streamlines)",
    )
    return p


def main():
    args = vars(build_argparser().parse_args())

    filepath = os.getcwd()

    name = None
    if args.get("folder_path"):
        name = args["folder_path"]
        print("Looking in folder", name)
        filepath = os.path.join(filepath, name)

    # determine number of subsets
    subsets = sum([x[-19:] == "_plausible_ref.json" for x in os.listdir(filepath)])

    if args.get("json_path"):
        json_path = args["json_path"]
        print("Analyzing only indices from", json_path)
        ind = get_indices_from_json(json_path)

    # get vote statistics for every of the 10M streamlines
    streamline_index = process_subsets(filepath)

    # call function that evaluates votes and writes them to [name].csv
    if args.get("json_path"):
        streamline_index = np.array(streamline_index[ind])
    evaluate_subsets(streamline_index, subsets, name)


if __name__ == "__main__":
    main()
