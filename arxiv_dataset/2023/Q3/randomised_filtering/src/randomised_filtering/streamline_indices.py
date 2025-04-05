"""Utilities for manipulating streamline indices."""

import numpy as np
import json

from typing import Tuple, List
from textwrap import dedent


def get_list_of_streamline_indices_from_mrtrix(
    path_to_mrtrix_selection_file: str,
) -> Tuple[List[int], List[int]]:
    """Read the mrtrix selection file and convert the 'binary mask'
    into streamline indices.

    Parameters
    ----------
    path_to_mrtrix_selection_file : str
        path to selection file created by tcksift with option -out_selection

    Returns
    -------
    idx_0 : List[int]
        list of streamline indices of 'implausible' streamlines
    idx_1 : List[int]
        list of streamline indices of 'plausible' streamlines
    """

    with open(path_to_mrtrix_selection_file, "r") as f:
        l: List[int] = [int(line.rstrip()) for line in f]

    arr = np.array(l)

    idx_0: np.ndarray = np.where(arr == 0)[0]
    idx_1: np.ndarray = np.where(arr == 1)[0]

    s = dedent(
        """
        Separation using mrtrix3 selection file:
            implausible: {nb_streamlines_0:>8}
            plausible  : {nb_streamlines_1:>8}
    """
    )
    print(s.format(nb_streamlines_0=idx_0.size, nb_streamlines_1=idx_1.size))

    return idx_0.tolist(), idx_1.tolist()


def write_list_of_streamline_indices(
    path_to_json_file: str, list_sl_idx: List[int], path_to_tractogram: str
) -> None:
    """Write given list of streamline indices to to json file.

    Parameters
    ----------
    path_to_json_file : str
        path to json file containing one list structure
    list_sl_idx : List[int]
        list of streamline indices
    path_to_tractogram : str
        path to tractogram file which the streamline indices are referring to
    """

    obj = {"filenames": [path_to_tractogram], path_to_tractogram: list_sl_idx}

    with open(path_to_json_file, "w") as f:
        json.dump(obj, f)
