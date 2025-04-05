#!/usr/bin/env python

import os

from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent
from tqdm import tqdm

from randomised_filtering.streamline_indices import (
    write_list_of_streamline_indices,
    get_list_of_streamline_indices_from_mrtrix,
)


DESC = """
Obtain the subsets of streamline indices of plausible and implausible
streamlines from streamline selection file of MRTrix3.
"""
EPILOG = dedent(
    """
    example calls:

      {filename} out_tractogram2_selection.txt out_tractogram2
    """.format(
        filename=os.path.basename(__file__)
    )
)


def build_parser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "selection_file",
        help="Path to mrtrix selection file containing 'binary mask' of selected "
        "streamlines (output of tcksift with option -out_selection).",
    )
    p.add_argument(
        "path_to_tractogram",
        help="Path to tractogram file which the streamline indices refer to.",
    )
    p.add_argument(
        "output_basename",
        help="Path to output file without file ending."
        " (The script will append 'plausible' or 'implausible' and the .json "
        "file ending.)",
    )
    return p


# SETTINGS
SUFFIX_PLAUSIBLE = "_plausible_indices"
SUFFIX_IMPLAUSIBLE = "_implausible_indices"
SUFFIX_FILE = ".json"


if __name__ == "__main__":
    args = vars(build_parser().parse_args())

    idx_implausible, idx_plausible = get_list_of_streamline_indices_from_mrtrix(
        args["selection_file"]
    )

    for suffix, idx_list in tqdm(
        list(
            zip(
                [SUFFIX_PLAUSIBLE, SUFFIX_IMPLAUSIBLE], [idx_plausible, idx_implausible]
            )
        ),
        desc="Writing index files",
    ):
        write_list_of_streamline_indices(
            args["output_basename"] + suffix + SUFFIX_FILE,
            idx_list,
            args["path_to_tractogram"],
        )
