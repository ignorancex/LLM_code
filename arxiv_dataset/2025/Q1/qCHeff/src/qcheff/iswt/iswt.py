# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from types import ModuleType

import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpsparse
import numpy as np
import scipy.sparse as spsparse

from qcheff.operators import (
    OperatorMatrix,
    SparseOperator,
    eye_like,
    qcheff_array,
    qcheff_dense_array,
    qcheff_sparse_array,
)


@dataclass
class ExactIterativeSWT:
    """
    Interface for the state machine encoding the iterative SWT algorithm.

    """

    H: OperatorMatrix
    givens_sparse_backend_: ModuleType | None = None
    copy: InitVar[bool] = field(default=True, init=True)

    def __post_init__(self, copy: bool = True) -> None:
        if isinstance(self.H, SparseOperator):
            self.givens_sparse_backend_ = self.H.backend_module
            self.H.op = self.H.op.tocsr(copy=copy)
        else:
            self.givens_sparse_backend_ = cupyx.scipy.get_array_module(self.H.op).sparse
            self.H.op = SparseOperator(self.H.op)

    @abstractmethod
    def givens_rotation_matrix(self, i: int, j: int) -> qcheff_sparse_array:
        """
        Returns a sparse Givens rotation matrix for the coupling (i,j).

        Parameters
        ----------
        i : int
            The first index of the coupling.
        j : int
            The second index of the coupling.

        Returns
        -------
        qcheff_sparse_array
            The sparse Givens rotation matrix.
        """

    def unitary_transformation(self, U: qcheff_array) -> None:
        """
        Apply a unitary transformation U to the Hamiltonian.

        Parameters
        ----------
        U : qcheff_array
            The unitary transformation matrix.
        """
        self.H.op = U @ self.H.op @ U.conj().T

    def eliminate_couplings(self, couplings: qcheff_dense_array) -> None:
        """
        Eliminate the coupling between levels i and j using a Givens Rotation.

        Parameters
        ----------
        couplings : qcheff_dense_array
            The array of couplings to eliminate.
        """
        Ul = eye_like(self.H.op)
        for i, j in couplings:
            Ul = self.givens_rotation_matrix(i, j) @ Ul

        self.unitary_transformation(U=Ul)

    def eliminate_coupling(self, i: int, j: int) -> None:
        """
        Eliminate the coupling between levels i and j using a Givens Rotation.

        Parameters
        ----------
        i : int
            The first index of the coupling.
        j : int
            The second index of the coupling.
        """
        self.unitary_transformation(U=self.givens_rotation_matrix(i, j))

    def largest_couplings(self, n: int = 1):
        """
        Mark the largest coupling in the Hamiltonian.

        Parameters
        ----------
        n : int, optional
            The number of largest couplings to return. Defaults to 1.

        Returns
        -------
        tuple
            The indices (i,j) of the largest coupling in the Hamiltonian.
        """
class NPADScipySparse(ExactIterativeSWT):
    """
    NPAD algorithm implemented with SciPy sparse matrices.

    Parameters
    ----------
    H : OperatorMatrix
        The Hamiltonian to be transformed.

    Attributes
    ----------
    H : OperatorMatrix
        The Hamiltonian to be transformed.

    Methods
    -------
    givens_rotation_matrix(i, j)
        Returns a sparse Givens rotation matrix for the coupling (i,j).
    largest_couplings(n=1, levels=None)
        Mark the largest coupling in the Hamiltonian.
    """

    def givens_rotation_matrix(self, i: int, j: int) -> qcheff_sparse_array:
        """
        Returns a sparse Givens rotation matrix for the coupling (i,j).

        Parameters
        ----------
        i : int
            The first index of the coupling.
        j : int
            The second index of the coupling.

        Returns
        -------
        qcheff_sparse_array
            The sparse Givens rotation matrix.
        """
        _H = self.H.op
        dim = _H.shape[0]
        Hi = _H[i, i]
        Hj = _H[j, j]
        Hij = _H[i, j]
        g = np.abs(Hij)

        if np.isclose(g, 0) or np.isclose(Hi, Hj):
            theta = 0
            phase = 1
        else:
            phase = Hij / g
            delta = np.real(0.5 * (Hi - Hj))
            inv_kappa = delta / g
            a1okappa = np.abs(inv_kappa)
            theta = np.arctan(
                np.sign(1 / inv_kappa) / (a1okappa + np.sqrt(1 + a1okappa * a1okappa))
            )

        c = np.cos(theta)
        s = np.sin(theta)

        diag_idx = np.linspace(0, dim - 1, dim, dtype=np.int32)
        row_idx = np.append(diag_idx, (i, j))
        col_idx = np.append(diag_idx, (j, i))
        # Empty array for all entries
        all_entries = np.ones(dim + 2, dtype=self.H.op.dtype)
        # replace with Givens rotation entries
        np.put(all_entries, (i, j, -2, -1), (c, c, s * phase, -s * np.conjugate(phase)))

        return spsparse.csr_array(
            (all_entries, (row_idx, col_idx)),
            shape=self.H.shape,
            dtype=self.H.op.dtype,
        )

    def largest_couplings(self, n: int = 1, levels=None):
        """
        Mark the largest coupling in the Hamiltonian.

        Parameters
        ----------
        n : int, optional
            The number of largest couplings to return. Defaults to 1.
        levels : array_like, optional
            The levels to consider when calculating the largest couplings.

        Yields
        ------
        tuple
            A tuple containing the indices (i,j) of the largest coupling and the
            Hamiltonian.
        """
        while True:
            coo_op = self.H.couplings().tocoo()
            num_couplings = min(n, coo_op.nnz)

            i, j, val = *coo_op.coords, coo_op.data

            if levels is not None:
                mask = np.logical_or(np.isin(i, levels), np.isin(j, levels))
                num_couplings = min(num_couplings, np.count_nonzero(mask))
                i, j, val = i[mask], j[mask], val[mask]
                if num_couplings == 1:
                    yield np.dstack((i, j)).squeeze(), self.H
            yield (
                np.dstack((i, j)).squeeze()[
                    np.abs(val).argpartition(-num_couplings)[-num_couplings:]
                ],
                self.H,
            )


class NPADCupySparse(ExactIterativeSWT):
    """
    NPAD algorithm implemented with CuPy sparse matrices.

    Parameters
    ----------
    H : OperatorMatrix
        The Hamiltonian to be transformed.

    Attributes
    ----------
    H : OperatorMatrix
        The Hamiltonian to be transformed.

    Methods
    -------
    givens_rotation_matrix(i, j)
        Returns a sparse Givens rotation matrix for the coupling (i,j).
    largest_couplings(n=1, levels=None)
        Mark the largest coupling in the Hamiltonian.
    """

    def givens_rotation_matrix(self, i: int, j: int) -> qcheff_sparse_array:
        """
        Returns a sparse Givens rotation matrix for the coupling (i,j).

        Parameters
        ----------
        i : int
            The first index of the coupling.
        j : int
            The second index of the coupling.

        Returns
        -------
        qcheff_sparse_array
            The sparse Givens rotation matrix.
        """
        _H = self.H.op
        dim = _H.shape[0]
        Hi = _H[i, i]
        Hj = _H[j, j]
        Hij = _H[i, j]
        g = cp.abs(Hij)

        if cp.isclose(g, 0) or cp.isclose(Hi, Hj):
            theta = 0
            phase = 1
        else:
            phase = Hij / g
            delta = cp.real(0.5 * (Hi - Hj))
            inv_kappa = delta / g
            a1okappa = cp.abs(inv_kappa)
            theta = cp.arctan(
                cp.sign(1 / inv_kappa) / (a1okappa + cp.sqrt(1 + a1okappa * a1okappa))
            )

        c = cp.cos(theta)
        s = cp.sin(theta)

        diag_idx = cp.linspace(0, dim - 1, dim, dtype=cp.int32)
        row_idx = cp.append(diag_idx, (i, j))
        col_idx = cp.append(diag_idx, (j, i))
        # Ones array for all entries
        all_entries = cp.ones(dim + 2, dtype=self.H.op.dtype)
        # replace with Givens rotation entries
        cp.put(all_entries, (i, j, -2, -1), (c, c, s * phase, -s * np.conjugate(phase)))

        return cpsparse.csr_matrix(
            (all_entries, (row_idx, col_idx)),
            shape=self.H.shape,
            dtype=self.H.op.dtype,
        )

    def largest_couplings(self, n: int = 1, levels=None):
        """
        Mark the largest coupling in the Hamiltonian.

        Parameters
        ----------
        n : int, optional
            The number of largest couplings to return. Defaults to 1.
        levels : array_like, optional
            The levels to consider when calculating the largest couplings.

        Yields
        ------
        tuple
            A tuple containing the indices (i,j) of the largest coupling and the
            Hamiltonian.
        """
        while True:
            coo_op = self.H.couplings().tocoo()
            num_couplings = min(n, coo_op.nnz)

            i, j, val = coo_op.row, coo_op.col, coo_op.data

            if levels is not None:
                levels = cp.array(levels)
                mask = np.logical_or(np.isin(i, levels), np.isin(j, levels))
                num_couplings = min(num_couplings, np.count_nonzero(mask))
                i, j, val = i[mask], j[mask], val[mask]
                if num_couplings == 1:
                    yield np.dstack((i, j)).squeeze(), self.H
            yield (
                cp.dstack((i, j)).squeeze()[
                    np.abs(val).argpartition(-num_couplings)[-num_couplings:]
                ],
                self.H,
            )
