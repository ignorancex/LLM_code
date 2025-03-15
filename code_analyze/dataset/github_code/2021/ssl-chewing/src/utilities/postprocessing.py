from typing import Tuple, Union

import numpy as np

from utilities.matlabutils import labels2intervals
from utilities.numpyutils import is_numpy_matrix, is_numpy_vector


def merge_gaps(intervals: np.ndarray, max_gap: float, return_durations: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Merge gaps smaller than ``max_gap``. Values of ``intervals`` must be in the same unit as ``max_gap`` (i.e. both
    should be in seconds, or both should be in samples).
    """
    assert is_numpy_matrix(intervals, cols=2)
    assert isinstance(max_gap, float)
    assert isinstance(return_durations, bool)

    intervals_duration = intervals[:, 1] - intervals[:, 0]
    merged = [intervals[0, :].tolist()]
    durations = [intervals_duration[0]]

    for i in range(1, intervals.shape[0]):
        if intervals[i, 0] - intervals[i - 1, 1] <= max_gap:
            merged[-1][1] = intervals[i, 1]
            durations[-1] += intervals_duration[i]
        else:
            merged.append(intervals[i, :].tolist())
            durations.append(intervals_duration[i])
    merged = np.array(merged)

    if return_durations:
        return merged, np.array(durations)
    else:
        return merged


def get_bouts(chewing: np.ndarray, fs_Hz: float, max_gap_sec: float = 2.0, min_duration_sec: float = 5.0) -> np.ndarray:
    """
    Computes a sequence of chewing-bout intervals from a boolean, chewing-indicator sequence.

    :param chewing: A sequence that indicates chewing (chewing = 1)
    :param fs_Hz: The sampling rate of the chewing sequence
    :param max_gap_sec: Maximum gap-duration that is merged between consecutive chews
    :param min_duration_sec: Minimum duration of a chewing bout
    :return: The 2-column (start & stop, in seconds) matrix of chewing bouts
    """
    assert is_numpy_vector(chewing)
    assert isinstance(fs_Hz, float)
    assert isinstance(max_gap_sec, float)
    assert isinstance(min_duration_sec, float)

    intervals, labels = labels2intervals(chewing)
    chews_sec = intervals[labels == 1, :] / fs_Hz

    merged_chews = merge_gaps(chews_sec, max_gap_sec)

    durations = merged_chews[:, 1] - merged_chews[:, 0]
    bouts_sec = merged_chews[durations >= min_duration_sec, :]

    return bouts_sec


def get_meals(bouts_sec: np.ndarray, max_gap_sec: float = 60.0, min_overlap: float = 0.25):
    """
    Computes a sequence of meal intervals from a sequence of chewing-bout intervals.

    :param bouts_sec: The sequence of chewing-bout intervals (see ``get_bouts`` output)
    :param max_gap_sec: Maximum gap-duration that is merged between consecutive chewing-bouts
    :param min_overlap: Minimum allowed overlap of chewing-bout duration with meal duration
    :return: The 2-column (start & stop, in seconds) matrix of meals
    """
    assert is_numpy_matrix(bouts_sec, cols=2)
    assert isinstance(max_gap_sec, float)
    assert isinstance(min_overlap, float)

    meals_sec, orig_durations_sec = merge_gaps(bouts_sec, max_gap_sec, True)
    overlap = orig_durations_sec / (meals_sec[:, 1] - meals_sec[:, 0])

    return meals_sec[overlap >= min_overlap, :]
