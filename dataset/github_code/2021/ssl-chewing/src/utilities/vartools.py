from typing import List

import numpy as np
from tabulate import tabulate

from utilities.matlabutils import find_unique
from utilities.numpyutils import is_numpy_vector
from utilities.printutils import get_tabulate_header


def label_histogram(y: np.ndarray, y_is_categorial: bool = False, labels2str: bool = True) -> (List, np.ndarray):
    """Prints a histogram of the labels, assuming integer labels."""
    assert is_numpy_vector(y)
    assert isinstance(y_is_categorial, bool)
    assert isinstance(labels2str, bool)

    if y_is_categorial:
        y = np.array([find_unique(y[i] == 1) for i in range(y.size)])

    u0: np.ndarray = np.unique(y)
    u: np.ndarray = np.hstack((u0, u0[-1] + 1)) - 0.5

    if labels2str:
        lbl = [str(x) for x in u0]
    else:
        lbl = u0.tolist()

    return lbl, np.histogram(y, u)[0]


def print_label_histogram(y: np.ndarray):
    assert is_numpy_vector(y)

    labels, hist_values = label_histogram(y)

    tbl: str = tabulate(hist_values.reshape((1,) + hist_values.shape), labels)
    hdr: str = get_tabulate_header(tbl, "CLASS HISTOGRAM", midrule=False)

    print(hdr + "\n" + tbl)
