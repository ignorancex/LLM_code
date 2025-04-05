#!/usr/bin/env python

import os
import json

from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent
from tqdm import tqdm
from typing import Tuple, Any

from randomised_filtering.streamline_indices import write_list_of_streamline_indices


DESC = """
Transform streamline indices from a streamline subset to another reference tractogram.
"""
EPILOG = dedent(
    """
    example calls:

      {filename} subset_indices.json subset_plausible.json subset_plausible_ref.json
    """.format(
        filename=os.path.basename(__file__)
    )
)


def build_parser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "subset_indices_ref", help="Path to index file with reference indices."
    )
    p.add_argument("subset_indices", help="Path to index file with subset indices.")
    p.add_argument("output_indices", help="Path to output index file.")
    return p


def read_index_list_from_file(json_file: str) -> Tuple[Any, Any]:
    """
    Read index list and reference tractogram filename from index file.

    Parameters
    ----------
    json_file : str
        path to json index file

    Returns
    -------
    path_to_ref_tractogram : str
        path to reference tractogram
    idx_list : Tuple[int, int]
        list of streamline indices in the reference tractogram
    """
    with open(json_file, "r") as f:
        idx_list = json.load(f)
        return idx_list["filenames"][0], idx_list[idx_list["filenames"][0]]


if __name__ == "__main__":
    args = vars(build_parser().parse_args())

    # load files
    ref_file, ref_idx = read_index_list_from_file(args["subset_indices_ref"])
    _, subset_idx = read_index_list_from_file(args["subset_indices"])

    # extract the indices in the reference streamline set
    subset_ref_idx = [
        ref_idx[i] for i in tqdm(subset_idx, desc="Extracting reference indices")
    ]

    # write reference indices
    write_list_of_streamline_indices(args["output_indices"], subset_ref_idx, ref_file)
