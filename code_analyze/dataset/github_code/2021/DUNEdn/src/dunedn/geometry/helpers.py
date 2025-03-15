"""
    This module contains the geometry helper functions that transform events into
    planes and vice versa.
"""
from typing import Tuple
import numpy as np
from dunedn.geometry.pdune import (
    nb_tdc_ticks,
    nb_ichannels,
    nb_apas,
    nb_apa_channels,
)


def evt2planes(event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts event array to planes.

    Parameters
    ----------
    event: np.array
        Raw Digit array, of shape=(nb_event_channels, nb_tdc_ticks).

    Returns
    -------
    collections: np.array
        Induction planes array, of shape=(N,C,H,W).
    collections: np.array
        Collection planes array, of shape=(N,C,H,W).
    """
    base = np.arange(nb_apas).reshape(-1, 1) * nb_apa_channels
    iidxs = [[0, nb_ichannels, 2 * nb_ichannels]] + base
    cidxs = [[2 * nb_ichannels, nb_apa_channels]] + base
    inductions = []
    for start, idx, end in iidxs:
        induction = [event[start:idx], event[idx:end]]
        inductions.extend(induction)
    collections = []
    for start, end in cidxs:
        collections.append(event[start:end])
    return np.stack(inductions)[:, None], np.stack(collections)[:, None]


def planes2evt(inductions: np.ndarray, collections: np.ndarray) -> np.ndarray:
    """
    Converts planes back to event.

    Parameters
    ----------
    inductions: np.array
        Induction planes, of shape=(N,C,H,W).
    collections: np.array
        Collection planes, of shape=(N,C,H,W).

    Returns
    -------
    np.array
        Raw Digits array, of shape=(nb_event_channels, nb_tdc_ticks).
    """
    inductions = np.array(inductions).reshape(-1, 2 * nb_ichannels, nb_tdc_ticks)
    collections = np.array(collections)[:, 0]
    event = []
    for i, c in zip(inductions, collections):
        event.extend([i, c])
    return np.concatenate(event)
