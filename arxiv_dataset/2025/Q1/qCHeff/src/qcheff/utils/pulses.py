# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from dataclasses import InitVar, dataclass

import cupy as cp
import numpy as np

__all__ = ["FourierPulse"]


def cos_ramp(t, a=0.2):
    """
    Compute a cosine ramp function.

    Parameters
    ----------
    t : array_like
        Time array.
    a : float, optional
        Ramp ratio. Default is 0.2.

    Returns
    -------
    array_like
        Cosine ramp function values.
    """
    xp = cp.get_array_module(t)

    return xp.piecewise(
        t,
        [
            xp.logical_and(0 <= t, t <= a),
            xp.logical_and(a <= t, t <= 1 - a),
            xp.logical_and(1 - a <= t, t <= 1),
        ],
        [
            lambda t: 0.5 * (1 + xp.cos(xp.pi * (t / a - 1))),
            1,
            lambda t: 0.5 * (1 + xp.cos(xp.pi * ((t - 1) / a - 1))),
        ],
    )


@dataclass(kw_only=True)
class ControlPulse:
    """
    Defines a control pulse.

    Contains an envelope in the form of PulseEnvelope or a derived class,
    carrier frequency and phase. The amplitude is assumed to be 1 if unspecified.

    The carrier is assumed to be a cosine wave.

    basis_funcs and ramp_func are defined so that the gate is active
    between t=0 and t=1. It is also assumed that they are all vectorized functions.

    Parameters
    ----------
    coeffs : list of float
        Coefficients for the basis functions.
    basis_funcs : list of Callable[[float], float]
        Basis functions.
    frequency : float, optional
        Carrier frequency. Default is 1.
    phase : float, optional
        Phase of the pulse. Default is 0.
    amplitude : float, optional
        Amplitude of the pulse. Default is 1.
    name : str or None, optional
        Name of the pulse. Default is None.

    Raises
    ------
    ValueError
        If the lengths of coeffs and basis_funcs are not the same.
    """

    coeffs: list[float]
    basis_funcs: list[Callable[[float], float]]
    frequency: float = 1.0
    phase: float = 0.0
    amplitude: float = 1.0
    name: InitVar[str | None] = None

    def __post_init__(self, name=""):
        if len(self.coeffs) != len(self.basis_funcs):
            msg = "The lengths of coeffs and basis_funcs must be the same."
            raise ValueError(msg)
        self.name = name

    def envelope(self, tlist):
        """
        Generates a pulse array given a timelist.

        Parameters
        ----------
        tlist : array_like
            List of time points.

        Returns
        -------
        array_like
            Pulse array.
        """
        pulse_array = sum(
            coeff * basis_func(tlist)
            for coeff, basis_func in zip(self.coeffs, self.basis_funcs, strict=False)
        )
        return pulse_array

    def __call__(self, tlist):
        """
        Generates a pulse array given a timelist.

        Parameters
        ----------
        tlist : array_like
            List of time points.

        Returns
        -------
        array_like
            Pulse array.
        """
        xp = cp.get_array_module(tlist)
        pulse_array = (
            self.amplitude
            * self.envelope(tlist)
            * xp.cos(2 * xp.pi * self.frequency * tlist + self.phase)
        )
        return pulse_array


class FourierPulse(ControlPulse):
    """
    Pulse in the modulated Fourier basis.
    This creates a Pulse with the basis functions given in the Fourier basis.

    The basis functions are:

    s(t/T) * cos(pi * n* t / gate_time) for even n
    s(t/T) * sin(pi * n* t / gate_time) for odd n

    and s(t/T) is the ramp function.

    where n is the index of the basis function. n starts from 1.

    Parameters
    ----------
    coeffs : list of float
        List of coefficients.
    gate_time : float
        Total time of Pulse. Used to scale the basis functions.
    ramp_ratio : float, optional
        Ratio of the ramp time to the total time. Default is 0.2.
    backend : str, optional
        "cpu" or "gpu". Default is "cpu".
    **kwargs : dict
        Additional keyword arguments to pass to ControlPulse.
    """

    def __init__(self, coeffs, gate_time, ramp_ratio=0.2, backend="cpu", **kwargs):
        """
        Initializes a FourierPulse object.

        Parameters
        ----------
        coeffs : list of float
            List of coefficients.
        gate_time : float
            Total time of Pulse. Used to scale the basis functions.
        ramp_ratio : float, optional
            Ratio of the ramp time to the total time. Default is 0.2.
        backend : str, optional
            "cpu" or "gpu". Default is "cpu".
        **kwargs : dict
            Additional keyword arguments to pass to ControlPulse.
        """

        def ramp(t):
            return cos_ramp(t, ramp_ratio)

        self.ramp_function = ramp
        xp = cp if backend == "gpu" else np
        basis_funcs = [
            lambda t, n=n: (
                ramp(t / self.gate_time) * xp.cos(np.pi * n * t / self.gate_time)
                if n % 2 == 0
                else ramp(t / self.gate_time) * xp.sin(xp.pi * n * t / self.gate_time)
            )
            for n in range(1, len(coeffs) + 1)
        ]
        super().__init__(coeffs=coeffs, basis_funcs=basis_funcs, **kwargs)
        self.gate_time = gate_time

    def envelope(self, tlist):
        """
        Generates a pulse array given a timelist.

        Parameters
        ----------
        tlist : array_like
            List of time points.

        Returns
        -------
        array_like
            Pulse array.
        """
        return super().envelope(tlist)

    def __call__(self, tlist):
        """
        Generates a pulse array given a timelist.

        Parameters
        ----------
        tlist : array_like
            List of time points.

        Returns
        -------
        array_like
            Pulse array.
        """
        pulse_array = super().__call__(tlist)
        return pulse_array

    def __str__(self):
        return (
            f"FourierPulseEnvelope(id={self.name}, "
            f"coeffs={self.coeffs}, gate_time={self.gate_time})"
        )

    def __repr__(self):
        return str(self)
