"""This module implements utility function for all the networks."""
from logging import Logger
from typing import Tuple
from collections.abc import Iterable
from time import time as tm
import numpy as np

supported_models = ["uscg", "cnn", "gcnn"]


class BatchProfiler:
    """Class to profile for loops steps.

    Useful to profile each batch prediction during DnModel inference.

    Example
    -------

    >>> from dunedn.networks.utils import BatchProfiler
    >>> from time import sleep
    >>> bp = BatchProfiler()
    >>> wrap = bp.set_iterable(range(10))
    >>> for i in wrap:
    ...     print(i)
    ...     sleep(0.01)
    >>> msg = bp.print_stats()
    >>> print(msg)
    """

    def __init__(self, drop_last=False):
        self.drop_last = drop_last

    def __iter__(self):
        self.times = []
        return self

    def __next__(self):
        self.times.append(tm())
        return next(self.iterable)

    @property
    def deltas(self) -> np.ndarray:
        """Computes the wall time intervals between steps.

        Sets the ``nb_batches`` attribute.

        Returns
        -------
        deltas: np.ndarray
            The result time intervals, of shape=(nb intervals,).
        """
        times = self.times[:-1] if self.drop_last else self.times
        deltas = np.diff(times)
        self.nb_batches = len(deltas)
        return deltas

    def get_stats(self) -> Tuple[float, float]:
        """Computes average and mean standard error on timings.

        Returns
        -------
        mean: float
            The average batch inference time.
        err: float
            The uncertainty on the batch inference step average time.
        """
        deltas = self.deltas
        sqrtn = np.sqrt(self.nb_batches)
        mean = deltas.mean()
        err = deltas.std() / sqrtn
        return mean.item(), err.item()

    def print_stats(self) -> str:
        """Human-readable message on profiled inference.

        Returns
        -------
        message: str
            The message with profiling information.
        """
        mean, std = self.get_stats()
        msg = (
            f"Forward pass with {self.nb_batches} batches. "
            f"Time per batch: {mean:.3e} +/- {std:.3e} s"
        )
        return msg

    def set_iterable(self, iterable: Iterable):
        """Sets the iterable to be profiled.

        Parameters
        ----------
        iterable
        """
        self.iterable = iter(iterable)
        return self


def get_supported_models():
    """Returns the names of the supported models.

    Returns
    -------
        - list, the list of currently implemented models
    """
    return supported_models


def get_hits_from_clear_images(planes: np.ndarray, threshold: float) -> np.ndarray:
    """Segment input images as signal-background pixels.

    Parameters
    ----------
    planes: np.ndarray
        The clear planes, of shape=(N,1,H,W).
    threshold: float
        Threshold above which a pixel is considered containing signal.

    Returns
    -------
    hits: np.ndarray
        The signal-background segmented image, of shape=(N,1,C,W).
    """
    mask = np.abs(planes) >= threshold
    hits = np.zeros_like(planes)
    hits[mask] = 1
    return hits


def apply_median_subtraction(planes: np.ndarray) -> np.ndarray:
    """Computes median subtraction to input planes.

    Parameters
    ----------
    planes: np.ndarray
        The data to be normalized.

    Returns
    -------
    output: np.ndarray
        The median subtracted data.
    """
    medians = np.median(planes, axis=[1, 2, 3], keepdims=True)
    output = planes - medians
    return output


def print_epoch_logs(logger: Logger, metrics_names: list[str], logs: dict):
    """Prints logs dictionary on epoch end.

    Parameters
    ----------
    logger: Logger
        The logging object.
    metrics_names: list[str]
        The list of metrics to be printed.
    logs: dict
        The computed metrics values to be logged.
    """
    msg = f"Took {logs['epoch_time']:.0f} s, "
    all_metrics_names = metrics_names + [f"val_{name}" for name in metrics_names]
    for name in all_metrics_names:
        mean = logs.get(name)
        std = logs.get(f"{name}_std")
        if mean is not None and std is not None:
            msg += f"{name}: {mean:.3f} +/- {std:.3f}, "
    msg = msg.strip(" ,")
    logger.info(msg)


def print_cfnm(cfnm, channel):
    """Prints confusion matrix.

    Parameters
    ----------
    cfnm: list
        Computed confusion matrix.
    channel: str
        Available options readout | collection.

    Returns
    -------
    msg: str
        The confusion matrix representatiton to be printed.
    """
    tp, fp, fn, tn = cfnm
    msg = (
        f"Confusion Matrix on {channel} planes:\n"
        f"\tTrue positives: {tp[0]:.3f} +- {tp[1]:.3f}\n"
        f"\tTrue negatives: {tn[0]:.3f} +- {tn[1]:.3f}\n"
        f"\tFalse positives: {fp[0]:.3f} +- {fp[1]:.3f}\n"
        f"\tFalse negatives: {fn[0]:.3f} +- {fn[1]:.3f}"
    )
    return msg
