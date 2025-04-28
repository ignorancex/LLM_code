from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import functools

import numpy as np

from src.lib.custom_vars import Ifm, Chn, NumpyPolynomial


@dataclass(frozen=True)
class Interferometer:
    wavenumber: np.ndarray[tuple[Chn], np.float64]
    opd: np.ndarray[tuple[Ifm, Chn], np.float64]
    reflectivity: np.ndarray[tuple[Ifm, Chn], np.float64]
    gain: np.ndarray[tuple[Ifm, Chn], np.float64]
    phase_shift: np.ndarray[tuple[Ifm, Chn], np.float64]

    @property
    def phase_difference(self):
        return phase_difference(
            opd=self.opd,
            wavenumber=self.wavenumber,
            phase_shift=self.phase_shift,
        )

    def transfer(
        self, model: int = 0, is_scaled: bool = False,
    ) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
        if is_scaled:
            return interferometry_transfer_scaled(
                wavenumber=self.wavenumber,
                opd=self.opd,
                reflectivity=self.reflectivity,
                phase_shift=self.phase_shift,
                gain=self.gain,
                model=model,
            )
        return interferometry_transfer(
            wavenumber=self.wavenumber,
            opd=self.opd,
            reflectivity=self.reflectivity,
            phase_shift=self.phase_shift,
            gain=self.gain,
            model=model,
        )


def nyquist_cutoff_opd(wavenumber: np.ndarray) -> float:
    """Returns the Shannon-Nyquist limit for the opd estimation"""
    return 1 / (2 * np.mean(np.diff(np.sort(wavenumber))).astype(float))


def phase_difference(
    wavenumber: np.ndarray[tuple[Chn], np.float64],
    opd: np.ndarray[tuple[Ifm, Chn], np.float64],
    phase_shift: np.ndarray[tuple[Ifm, Chn], np.float64],
) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
    """Evaluates the phase difference between optical rays, given the opd"""
    return 2 * np.pi * wavenumber[np.newaxis, :] * opd - phase_shift


def interferometry_transfer_scaled(
    wavenumber: np.ndarray[tuple[Chn], np.float64],
    opd: np.ndarray[tuple[Ifm, Chn], np.float64],
    reflectivity: np.ndarray[tuple[Ifm, Chn], np.float64],
    phase_shift: np.ndarray[tuple[Ifm, Chn], np.float64],
    gain: np.ndarray[tuple[Ifm, Chn], np.float64],
    model: int = 0,
) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
    """Alternative transfer function, with mean equal to the gain."""
    if model == 0:
        gain = gain * (1 - reflectivity) * (1 + reflectivity)
    elif model == 1:
        gain = gain
    elif model == 2:
        gain = gain / (1 + reflectivity ** 2)
    else:
        gain = gain * (1 - reflectivity ** 2)
        gain = gain / (1 + reflectivity ** (2 * model))
    return interferometry_transfer(
        wavenumber=wavenumber,
        opd=opd,
        reflectivity=reflectivity,
        phase_shift=phase_shift,
        gain=gain,
        model=model,
    )


def interferometry_transfer(
    wavenumber: np.ndarray[tuple[Chn], np.float64],
    opd: np.ndarray[tuple[Ifm, Chn], np.float64],
    reflectivity: np.ndarray[tuple[Ifm, Chn], np.float64],
    phase_shift: np.ndarray[tuple[Ifm, Chn], np.float64],
    gain: np.ndarray[tuple[Ifm, Chn], np.float64],
    model: int = 0,
) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
    """
    Calculates the transfer function from the given parameters.
    wavenumber is a 1d array of the wavenumber samples.
    opd, gain, reflectivity and phase_shift are same-shaped 2d arrays and their
    first column must match the length of the wavenumber array.
    model is the interferometer transfer model (0: Airy's distribution, any
    positive integer N: N-wave model)
    This is the main function for the calculation of the transfer function;
    different parameters formats have to interface here
    """

    phase = phase_difference(
        opd=opd, wavenumber=wavenumber, phase_shift=phase_shift
    )
    if model == 0:
        denominator_dc = (1 - reflectivity) ** 2
        denominator_ac = 4 * reflectivity * np.sin(phase / 2) ** 2
        transfer = 1 / (denominator_dc + denominator_ac)
    elif model == 1:
        transfer = 1
    elif model == 2:
        transfer = 1 + reflectivity**2 + 2 * reflectivity * np.cos(phase)
    else:
        numerator_dc = 1 + reflectivity ** (2 * model)
        numerator_ac = -2 * reflectivity ** model * np.cos(model * phase)
        denominator = 1 + reflectivity ** 2 - 2 * reflectivity * np.cos(phase)
        transfer = (numerator_dc + numerator_ac) / denominator
    return transfer * gain


def poly_eval(
    coef: np.ndarray[tuple[Any], np.float_],
    func: NumpyPolynomial,
    x: np.ndarray[tuple[Any], np.float_],
) -> np.ndarray:
    """Evaluates a parametric function (eg: numpy polynomial) in x"""
    return func(coef)(x)


def poly_eval_2d(
    coef: np.ndarray[tuple[Ifm, Any], np.float64],
    func: NumpyPolynomial,
    x: np.ndarray[tuple[Chn], np.float64],
) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
    partial = functools.partial(poly_eval, func=func, x=x)
    return np.apply_along_axis(func1d=partial, axis=1, arr=coef)
