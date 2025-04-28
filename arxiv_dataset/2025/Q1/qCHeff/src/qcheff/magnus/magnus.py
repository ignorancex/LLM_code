# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import scipy.linalg as la

from qcheff.magnus.utils_gpu import expm_taylor
from qcheff.operators import OperatorMatrix, qcheff_array
from qcheff.operators.sparse_operator import SparseOperator


@dataclass(kw_only=True)
class MagnusTimeEvol:
    tlist: qcheff_array
    drift_ham: OperatorMatrix
    control_sigs: qcheff_array
    control_hams: Sequence[OperatorMatrix]
    dims: tuple[int, int] = field(init=False)
    dt: float = field(init=False)
    tlims: tuple[float, float] = field(init=False)
    num_controls: int = field(init=False)
    expm: Callable = field(init=False)
    _magnus_tlist: qcheff_array | None = field(init=False, default=None)
    _magnus_hams: qcheff_array | None = field(init=False, default=None)
    _magnus_props: qcheff_array | None = field(init=False, default=None)

    def __post_init__(self):
        self.dims = self.drift_ham.op.shape

        num_control_sigs = (
            1 if len(self.control_sigs.shape) == 1 else self.control_sigs.shape[0]
        )

        if num_control_sigs != len(self.control_hams):
            msg = f"The number of control signals ({len(self.control_sigs)})\
                    and control Hamiltonians ({len(self.control_hams)}) must \
                        be the same."
            raise ValueError(msg)
        self.num_controls = num_control_sigs
        num_tpts = self.tlist.shape[0]
        t0, tf = float(self.tlist[0]), float(self.tlist[-1])
        self.dt = float((tf - t0) / (num_tpts - 1))
        self.tlims = float(t0), float(tf)
        self.expm = (
            expm_taylor if self.drift_ham.backend_module in [cp, cpsparse] else la.expm
        )

    def update_control_sigs(self, control_sigs: qcheff_array) -> None:
        """
        Update the control_sigs array.

        Parameters
        ----------
        control_sigs : np.ndarray or cp.ndarray
            The new control_sigs array. Must have the same shape as the current
            control_sigs array.

        Raises
        ------
        ValueError
            If the shape of the new array does not match the current array.
        """
        if control_sigs.shape != self.control_sigs.shape:
            msg = f"The new control_sigs array must have the \
                    same shape as the current control_sigs array. \
                    New shape: {control_sigs.shape}, \
                    Current shape: {self.control_sigs.shape}"
            raise ValueError(msg)
        self.control_sigs = control_sigs

    @abstractmethod
    def magnus_hamiltonians(
        self,
        **kwargs,
    ) -> qcheff_array | Generator[qcheff_array, None, None]:
        pass

    @abstractmethod
    def magnus_propagators(
        self,
        **kwargs,
    ) -> qcheff_array | Generator[qcheff_array, None, None]:
        pass

    def evolve(
        self,
        init_state: qcheff_array,
        **kwargs,
    ) -> Generator[qcheff_array, None, None]:
        """
        Evolve the initial state using the Magnus expansion.

        Parameters
        ----------
        init_state : qcheff_array
            The initial state to evolve.
        **kwargs : dict
            Additional keyword arguments for the evolution.

        Yields
        ------
        qcheff_array
            The state at the end of each interval.
        """
        xp = cp.get_array_module(self.drift_ham.op)
        current_state = np.ravel(xp.asarray(init_state))  # Forces a copy

        for prop in self.magnus_propagators(**kwargs):
            yield (current_state := prop @ current_state)


@dataclass(kw_only=True)
class MagnusTimeEvolDense(MagnusTimeEvol):
    def magnus_hamiltonians(
        self,
        **kwargs,
    ) -> qcheff_array:
        """
        Compute the Magnus Hamiltonians.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the computation.

        Returns
        -------
        qcheff_array
            The computed Magnus Hamiltonians.

        Raises
        ------
        ValueError
            If both or neither of num_intervals and points_per_interval are provided.
        """
        _num_intervals = kwargs.get("num_intervals", None)
        _points_per_interval = kwargs.get("points_per_interval", None)

        if bool(_num_intervals) == bool(_points_per_interval):  # XOR
            msg = "Either num_intervals or points_per_interval must be provided."
            raise ValueError(msg)
        elif _num_intervals is not None:
            _points_per_interval = int(np.ceil(len(self.tlist) / _num_intervals))
        elif _points_per_interval is not None:
            _num_intervals = int(np.ceil(len(self.tlist) / _points_per_interval))
        else:
            msg = "Something went wrong."
            raise ValueError(msg)
        # First, check if hamiltonians have already been computed.
        # If they have, check if the dimensions are commensurate with
        # the number of intervals and points per interval.

        if self._magnus_hams is not None:
            if self._magnus_hams.shape[1] == _num_intervals:
                return self._magnus_hams

        # Get the number of control signals

        # Get all the control signals sampled at the time list and reshape
        # into Magnus intervals. Only recompute if the number
        # of intervals has changed.

        xp = cp.get_array_module(self.drift_ham.op)

        control_ham_arr = xp.array([x.op for x in self.control_hams])
        magnus1_ctrl_coeffs = (
            self.control_sigs.reshape(
                (self.num_controls, _num_intervals, _points_per_interval)
            )
            .sum(axis=-1)
            .T
        )
        self._magnus_hams = self.dt * (
            _points_per_interval * self.drift_ham.op[None, ...]
            + np.einsum("ni,ijk->njk", magnus1_ctrl_coeffs, control_ham_arr)
        )

        return self._magnus_hams

    def magnus_propagators(
        self,
        **kwargs,
    ) -> qcheff_array:
        """
        Compute the Magnus propagators.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the computation.

        Returns
        -------
        qcheff_array
            The computed Magnus propagators.
        """
        return (
            self.expm(-1j * self.magnus_hamiltonians(**kwargs))
            if self._magnus_props is None
            else self._magnus_props
        )


@dataclass(kw_only=True)
class MagnusTimeEvolSparseLazy(MagnusTimeEvol):
    """
    Sparse version of MagnusTimeEvol. This version uses a lazy approach.
    """

    tlist: qcheff_array
    drift_ham: SparseOperator
    control_sigs: qcheff_array
    control_hams: Sequence[SparseOperator]

    def magnus_hamiltonians(
        self,
        **kwargs,
    ) -> Generator[SparseOperator, None, None]:
        """
        Compute the Magnus Hamiltonians lazily.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the computation.

        Yields
        ------
        qcheff_array
            The computed Magnus Hamiltonians.

        Raises
        ------
        ValueError
            If both or neither of num_intervals and points_per_interval are provided.
        """
        _num_intervals = kwargs.get("num_intervals", None)
        _points_per_interval = kwargs.get("points_per_interval", None)

        if bool(_num_intervals) == bool(_points_per_interval):  # XOR
            msg = "Either num_intervals or points_per_interval must be provided."
            raise ValueError(msg)
        elif _num_intervals is not None:
            _points_per_interval = int(np.ceil(len(self.tlist) / _num_intervals))
        elif _points_per_interval is not None:
            _num_intervals = int(np.ceil(len(self.tlist) / _points_per_interval))
        else:
            msg = "Something went wrong."
            raise ValueError(msg)

        cp.get_array_module(self.drift_ham.op)
        # Reshape into Magnus intervals.
        magnus1_ctrl_coeffs = self.dt * (
            self.control_sigs.reshape(
                (self.num_controls, _num_intervals, _points_per_interval)
            )
            .sum(axis=-1)
            .T
        )

        magnus1_drift_ham = self.dt * _points_per_interval * self.drift_ham.op

        for coeffs in magnus1_ctrl_coeffs:
            yield magnus1_drift_ham + sum(
                x[0] * x[1].op for x in (zip(coeffs, self.control_hams, strict=True))
            )

    def magnus_propagators(
        self,
        **kwargs,
    ) -> Generator[qcheff_array, None, None]:
        """
        Compute the Magnus propagators lazily.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the computation.

        Yields
        ------
        qcheff_array
            The computed Magnus propagators.
        """
        for magnus_interval_ham in self.magnus_hamiltonians(**kwargs):
            # Since there is no sparse version for expm, we use the dense version
            magnus_interval_prop = self.expm(-1j * magnus_interval_ham.toarray())
            yield magnus_interval_prop
