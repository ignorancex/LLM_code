from typing import Optional

import numpy as np
import pandas as pd

from plato.instrument.detection import DetectionModel


def compute_detectable_fraction(
    targets: pd.DataFrame,
    detection_model: DetectionModel,
    detection_threshold: float,
    axis: Optional[int | tuple[int]] = None,
) -> np.ndarray:
    """
    Calculate the fraction of detectable targets for a given sample.

    Parameters
    ----------
    targets : pd.DataFrame
        The target dataframe, must contain all
        necessary columns for detection_model.detection_efficiency.
        See DetectionModel for more details.
    detection_model : DetectionModel
        The detection model instance.
    detection_threshold : float
        The detection threshold, i.e., the minimum
        detection efficiency required to consider a target
        as detectable.
    axis : Optional[int  |  tuple[int]], optional
        Axis along which to compute the mean. If None,
        the mean is computed over all axes. By default, None.

    Returns
    -------
    np.ndarray
        The fraction of detectable targets.
    """

    efficiency = detection_model.detection_efficiency(targets)
    return np.mean(efficiency > detection_threshold, axis=axis)


def iterate_detectable_fraction(
    properties: pd.DataFrame,
    target_stars: pd.DataFrame,
    detection_model: DetectionModel,
    detection_threshold: float,
    reshaping_bins: Optional[int | tuple[int, int]] = None,
) -> np.ndarray:
    """
    Compute the detectable fraction for a list of targets,
    with varying properties. The target stars are assigned
    the properties from the properties dataframe, and the
    detectable fraction is calculated for each set of
    properties. This is done iteratively for all rows in
    the properties dataframe. Useful if both properties
    and target_stars are long so that properties do not fit
    into a single dataframe.

    Parameters
    ----------
    properties : pd.DataFrame
        The dataframe containing the properties to iterate over.
        The columns of this dataframe are assigned to the target_stars
        dataframe, and the detectable fraction is calculated for each
        row.
    target_stars : pd.DataFrame
        The target stars dataframe. Between this and the properties
        dataframe, all necessary columns for
        detection_model.detection_efficiency must be included. See
        DetectionModel for more details.
    detection_model : DetectionModel
        The detection model instance.
    detection_threshold : float
        The detection threshold, i.e., the minimum
        detection efficiency required to consider a target
        as detectable.
    reshaping_bins : Optional[int  |  tuple[int, int]], optional
        If not None, the resulting array is reshaped to the
        given shape. If an integer is given, the resulting
        array is reshaped to a square array with the given
        integer as the side length. By default, None.

    Returns
    -------
    np.ndarray
        The detectable fraction for each set of properties.
    """

    results: list = []
    for _, row in properties.iterrows():
        targets = target_stars.assign(**{col: row[col] for col in properties.columns})

        frac = compute_detectable_fraction(
            targets,
            detection_model=detection_model,
            detection_threshold=detection_threshold,
        )
        results.append(frac)

    reshaping_bins = (
        (reshaping_bins, reshaping_bins)
        if isinstance(reshaping_bins, int)
        else reshaping_bins
    )
    result_arr = np.array(results)

    if reshaping_bins is not None:
        assert isinstance(reshaping_bins, tuple)
        result_arr = result_arr.reshape(reshaping_bins)
    return result_arr
