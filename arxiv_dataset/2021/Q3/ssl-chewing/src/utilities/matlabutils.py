from typing import List, Union, Optional

import numpy as np

from utilities.numpyutils import is_numpy_vector
from utilities.typingutils import is_typed_list


def strcmp(lst: List[str], s: str) -> List[bool]:
    assert is_typed_list(lst, str)
    assert isinstance(s, str)

    return [x == s for x in lst]


def find(b: Union[List[bool], np.ndarray], n: Optional[int] = None) -> List[int]:
    assert is_typed_list(b, bool) or is_numpy_vector(b)
    assert isinstance(n, int) or n is None

    if n is None:
        # Quick and easy
        return [i for i, bi in enumerate(b) if bi]

    # n is not None
    idxs: List[int] = []
    count: int = 0
    for i, bi in enumerate(b):
        if bi:
            idxs.append(i)
            count += 1
            if count == n:
                break

    return idxs


def find_unique(b: List[bool]) -> int:
    idx: List[int] = find(b)

    if len(idx) == 1:
        return idx[0]
    else:
        raise ValueError("Cannot find unique (sum(b)=" + str(sum(b)) + ")")


def integer2path(n: int, max_files: int, files_ord: int = 3, folders_ord: int = 2) -> str:
    # Max digits
    k = len(str(max_files - 1))

    s = str(n).zfill(k)

    if len(s) <= files_ord:
        return s

    file_part = s[-files_ord:]
    folders_part = s[:-files_ord]

    p = len(folders_part) - files_ord
    while p > 0:
        folders_part = folders_part[:p] + '/' + folders_part[:p]
        p = p - folders_ord

    return folders_part + '/' + file_part


def labels2intervals(labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Convert a sequence of labels to a matrix of intervals.

    :param labels: The labels sequence
    :return: A two-column (start-stop index) matrix of the intervals
    """
    assert is_numpy_vector(labels)

    unset: int = -1
    n: int = labels.size

    # Initialize states
    intervals = [[0, unset]]
    intervals_labels = [labels[0]]

    # Loop labels
    for i in range(1, n):
        if labels[i] == labels[i - 1]:
            # Same label, continue
            continue
        intervals[-1][1] = i - 1
        intervals.append([i, unset])
        intervals_labels.append(labels[i])

    intervals[-1][1] = n

    return np.array(intervals), np.array(intervals_labels)
