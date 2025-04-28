# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import importlib
from collections.abc import Sequence
from dataclasses import dataclass

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.sparse as spsparse

from qcheff.magnus import magnus
from qcheff.operators import DenseOperator, SparseOperator
from qcheff.operators.operator_base import OperatorMatrix
from qcheff.utils.pulses import ControlPulse


############################################################################
# The QuTiPSystem class is a dataclass that describes a quantum system.
# This is here for convenience. Not part of the numerical implementations
# of the Magnus expansion. Do not optimize.
############################################################################
@dataclass(frozen=True)
class QuTiPSystem:
    """
    Basic system description.

    Attributes
    ----------
    drift_ham : qt.Qobj or None
        Drift Hamiltonian. Qobj for the static part of the Hamiltonian.
    control_sigs : list of ControlPulse
        Control signals given as a list of ControlPulse objects.
        These encode the time-dependent part of the Hamiltonian.
    control_hams : list of qt.Qobj
        Time independent Qobj for the control Hamiltonians.
        These, when multiplied by the control signals, give the time-dependent part of
        the Hamiltonian.
    """

    drift_ham: qt.Qobj | None
    control_sigs: Sequence[ControlPulse]
    control_hams: Sequence[qt.Qobj]

    def __post_init__(self):
        """
        Post-initialization checks for the QuTiPSystem class.

        Raises
        ------
        ValueError
            If the lengths of control_sigs and control_hams are not the same.
            If drift_ham is not Hermitian.
            If any of the control_hams are not Hermitian.
        """
        if len(self.control_sigs) != len(self.control_hams):
            msg = "The lengths of control_sigs and control_hams must be the same."
            raise ValueError(msg)

        # Only check if drift ham is provided.
        if self.drift_ham is not None:
            if not self.drift_ham.isherm:
                msg = "drift_ham must be Hermitian."
                raise ValueError(msg)

        for i, ham in enumerate(self.control_hams):
            if not ham.isherm:
                msg = f"control_hams[{i}]: {ham} is not Hermitian."
                raise ValueError(msg)

    def get_qutip_tdham(self, tlist):
        """
        Returns a dict that works with qutip.mesolve.
        Only written for compatibility with QuTiP.
        This will not be used in Magnus.

        Parameters
        ----------
        tlist : array_like
            List of time points.

        Returns
        -------
        list
            List of Hamiltonians and their corresponding time-dependent coefficients.
        """
        ham_t = [self.drift_ham] if self.drift_ham is not None else []
        for sig, ham in zip(self.control_sigs, self.control_hams, strict=True):
            ham_t.append([ham, sig(tlist)])
        return ham_t

    def get_magnus_system(self, tlist, *, device="cpu", sparse: bool = False):
        """
        Returns a MagnusTimeEvol object, using the correct backend for CPU/GPU.
        The backend is chosen based on the device argument.
        This is used to set up the Magnus time-evolution object.
        Not meant for optimization.

        Parameters
        ----------
        tlist : array_like
            List of time points.
        device : str, optional
            Device to use, either 'cpu' or 'gpu'. Default is 'cpu'.
        sparse : bool, optional
            Whether to use sparse matrices. Default is False.

        Returns
        -------
        magnus.MagnusTimeEvol
            Magnus time-evolution object.

        Raises
        ------
        ValueError
            If sparse is not True or False.
            If device is not 'cpu' or 'gpu'.
        """
        if sparse not in [True, False]:
            msg = "sparse must be True or False"
            raise ValueError(msg)

        if device not in ["cpu", "gpu"]:
            msg = "device must be 'cpu' or 'gpu'"
            raise ValueError(msg)

        xp = importlib.import_module("cupy" if device == "gpu" else "numpy")

        operator_type: type[OperatorMatrix]
        if sparse is True:
            operator_type = SparseOperator
            magnus_drift_ham = spsparse.csr_array(self.drift_ham[:])  # type: ignore
            magnus_control_hams = [spsparse.csr_array(x[:]) for x in self.control_hams]
            if device == "gpu":
                magnus_drift_ham = cpsparse.csr_matrix(magnus_drift_ham)
                magnus_control_hams = [
                    cpsparse.csr_matrix(x) for x in magnus_control_hams
                ]
        elif sparse is False:
            operator_type = DenseOperator
            magnus_drift_ham = xp.asarray(self.drift_ham[:])  # type: ignore
            magnus_control_hams = [DenseOperator(xp.asarray(x[:])) for x in self.control_hams]
            # if device == "gpu":
            #     magnus_drift_ham = cp.asarray(magnus_drift_ham)
            #     magnus_control_hams = [
            #         DenseOperator(cp.asarray(x)) for x in magnus_control_hams
            #     ]

        magnus_drift_ham = operator_type(magnus_drift_ham)

        # asnumpy is used since cp.piecewise does not support callable functions.
        control_sigs = xp.vstack(
            xp.asarray([sig(cp.asnumpy(tlist)) for sig in self.control_sigs])
        )
        return magnus(
            tlist=tlist,
            drift_ham=magnus_drift_ham,
            control_sigs=control_sigs,
            control_hams=magnus_control_hams,
        )

    def plot_control_signals(self, tlist, axis=None, **kwargs):  # pragma: no cover
        """
        Plotting pulses and such.

        Parameters
        ----------
        tlist : array_like
            List of time points.
        axis : matplotlib.axes.Axes, optional
            Matplotlib axis object. If None, a new figure and axis will be created.

        Returns
        -------
        None
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, layout="constrained")
        for sig in self.control_sigs:
            axis.plot(tlist, sig(tlist), label=sig.name, **kwargs)
            axis.set(ylim=(-1, 1))
            axis.set_ylabel("Amplitude")
        axis.legend()
