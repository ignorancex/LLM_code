"""
General purpose utilities
"""


from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.common_utils.custom_vars import Opd, Wvn, Acq, Deg


def add_noise(
        array: np.ndarray[tuple[Opd | Wvn, Acq], np.dtype[np.float_]],
        snr_db: float,
) -> np.ndarray[tuple[Opd | Wvn, Acq], np.dtype[np.float_]]:
    snr_rms = 10 ** (snr_db / 20)
    noise_normal = np.random.randn(*array.shape)
    signal_std = array.std(axis=-2, keepdims=True)
    alpha = signal_std / snr_rms
    noise = alpha * noise_normal
    return array + noise


def center(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_mean: float = 0.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Subtract the mean of an array, e.g., array_centered = array - mean"""
    array_mean = array.mean(axis=axis, keepdims=True)
    array_centered = array - array_mean
    if new_mean != 0.:
        array_centered = array_centered + new_mean
    return array_centered


def rescale(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_max: float = 1.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Rescale an array to a new maximum, e.g., array_normed = array / max"""
    array_max = array.max(axis=axis, keepdims=True)
    array_normalized = array / array_max
    if new_max != 1.:
        array_normalized = array_normalized * new_max
    return array_normalized


def min_max_normalize(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_min: float = 0.,
        new_max: float = 1.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Normalize an array to be in the range [0, 1], e.g., array_normed = (array - min) / (max - min)"""
    array_min = array.min(axis=axis, keepdims=True)
    array_normalized = (array - array_min) / (array.max(axis=axis, keepdims=True) - array_min)
    if new_min != 0. or new_max != 1.:
        array_normalized = new_min + array_normalized * (new_max - new_min)
    return array_normalized


def standardize(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_mean: float = 0.,
        new_std: float = 1.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Standardize an array, e.g., array_std = (array - mean) / std."""
    array_mean = array.mean(axis=axis, keepdims=True)
    array_std = array.std(axis=axis, keepdims=True)
    array_standardized = (array - array_mean) / array_std
    if np.any(new_mean != 0) or np.any(new_std != 1.):
        array_standardized = new_std * array_standardized + new_mean
    return array_standardized


def match_stats(
        array: np.ndarray[..., np.dtype[np.float_]],
        reference: np.ndarray[..., np.dtype[np.float_]],
        axis: int = -2,
        is_rescale_reference: bool = False,
) -> tuple[np.ndarray[..., np.dtype[np.float_]], np.ndarray[..., np.dtype[np.float_]]]:
    """
    Used mostly for plot purposes, especially when comparing arrays.
    1- Rescale the reference if needed, to a maximum of 1
    2- Standardize the array to the statistics of the rescaled reference
    """
    if is_rescale_reference:
        reference = rescale(array=reference, axis=axis)
    ref_mean = reference.mean(axis=axis, keepdims=True)
    ref_std = reference.std(axis=axis, keepdims=True)
    array_standardized = standardize(array=array, new_mean=ref_mean, new_std=ref_std, axis=axis)
    return array_standardized, reference


def calculate_rmse(
        array: np.ndarray[..., np.dtype[np.float_]],
        reference: np.ndarray[..., np.dtype[np.float_]],
        is_match_stats: bool = False,
        is_rescale_reference: bool = False,
        is_match_axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """
    Calculate Normalized Root Mean Squared Error.
    """
    if is_match_stats:
        array, reference = match_stats(
            array=array,
            reference=reference,
            axis=is_match_axis,
            is_rescale_reference=is_rescale_reference,
        )
    error = array - reference
    error_vectorized = error.reshape(*error.shape[0:array.ndim-reference.ndim], -1)
    error_norm = np.linalg.norm(x=error_vectorized, ord=2, axis=-1)
    reference_norm = np.linalg.norm(x=reference)
    rmse = error_norm / reference_norm
    return rmse


def generate_sampled_opds(nb_opd: int, opd_step: float, opd_min: float = 0) -> np.ndarray[tuple[Opd], np.dtype[np.float_]]:
    opds = opd_step * np.arange(nb_opd) + opd_min
    return opds


def generate_wavenumbers_from_opds(
        wavenumbers_num: int,
        del_opd: float,
        wavenumbers_start: float = None,
        wavenumbers_stop: float = None,
) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
    del_wn = 1 / (2 * wavenumbers_num * del_opd)  # del_wn tends to zero as nb_wn tends to infinity (implies continuous)
    wavenumbers = del_wn * (np.arange(wavenumbers_num) + 1/2)
    if wavenumbers_start is not None:
        wavenumbers = wavenumbers[wavenumbers >= wavenumbers_start]
    if wavenumbers_stop is not None:
        wavenumbers = wavenumbers[wavenumbers <= wavenumbers_stop]
    return wavenumbers


def convert_meter_units(values: float | np.ndarray, from_: str, to_: str):
    unit_to_meters = {
        "m": 1,
        "cm": 1e-2,
        "um": 1e-6,
        "nm": 1e-9,
    }

    if from_ not in unit_to_meters:
        raise ValueError(f"Unit {from_} is not supported")
    if to_ not in unit_to_meters:
        raise ValueError(f"Unit {to_} is not supported")

    value_in_meters = values * unit_to_meters[from_]
    converted_values = value_in_meters / unit_to_meters[to_]
    return converted_values


def convert_hertz_to_meter(values: float | np.ndarray, to_: str = "m"):
    light_speed = 3 * 10**8  # m/s = m*Hz
    light_speed = convert_meter_units(light_speed, "m", to_)
    converted_values = light_speed / values
    return converted_values


def polyval_rows(
        coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
        interval: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
    powers = np.arange(coefficients.shape[1])[:, None]
    interval_powered = np.power(interval, powers)
    polynomials = coefficients @ interval_powered
    return polynomials


def numpy_to_dataframe(
        array: np.ndarray,
        row_labels: list[str],
):
    df = pd.DataFrame(array, index=row_labels)
    return df


def numpy_to_latex(
        array: np.ndarray,
        row_labels: list[str],
        header: bool | list[str] = True,
        index: bool = True,
        na_rep: str = "Nan",
        float_format: str = None,
        caption: str = "",
        position: str = "h",
) -> str:
    df = pd.DataFrame(array, index=row_labels)
    latex_table = df.to_latex(
        header=header,
        index=index,
        na_rep=na_rep,
        float_format=float_format,
        caption=caption,
        position=position,
    )
    return latex_table


def convert_zero_to_infty_latex(order: int) -> str:
    if order == 0:
        return r"$\infty$"
    else:
        return f"{order:.0f}"


def oversample(array: np.ndarray, factor: int):
    original_points = len(array)
    new_points = original_points * factor
    original_indices = np.arange(original_points)
    new_indices = np.linspace(0, original_points - 1, new_points)
    interp_func = interp1d(original_indices, array, kind='linear')
    oversampled_array = interp_func(new_indices)
    return oversampled_array
