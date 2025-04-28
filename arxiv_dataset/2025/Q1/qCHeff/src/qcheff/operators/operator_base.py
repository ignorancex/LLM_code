# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from dataclasses import dataclass, field
from types import ModuleType
from typing import (
    Any,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import scipy.sparse as spsparse

qcheff_dense_array: TypeAlias = np.ndarray | cp.ndarray
qcheff_sparse_array: TypeAlias = cpsparse.spmatrix | spsparse.sparray
qcheff_array: TypeAlias = qcheff_dense_array | qcheff_sparse_array

SparseMatrix: TypeAlias = spsparse.csr_array | cpsparse.csr_matrix
DenseMatrix: TypeAlias = np.ndarray | cp.ndarray


@runtime_checkable
@dataclass
class OperatorMatrix(Protocol):
    """
    Interface for all operators used in qcheff.

    Parameters
    ----------
    op : qcheff_array
        The underlying array of the operator.
    backend_module : ModuleType, optional
        The module providing the backend for the operator. Defaults to
        the module of the input operator.

    """

    op: qcheff_array
    backend_module: ModuleType = field(init=False)

    @abstractmethod
    def save(self, filename: str) -> None:
        """
        Save the OperatorMatrix to disk.

        Parameters
        ----------
        filename : str
            The filename to save to.

        """

    @classmethod
    @abstractmethod
    def load(cls, filename: str) -> "OperatorMatrix":
        """
        Load an OperatorMatrix from disk.

        Parameters
        ----------
        filename : str
            The filename to load from.

        Returns
        -------
        OperatorMatrix
            The loaded OperatorMatrix.
        """

    def diagonals(self) -> qcheff_array:
        """
        Returns the diagonal elements of the operator.

        Returns
        -------
        qcheff_array
            The diagonal elements of the operator.
        """

    def couplings(self) -> qcheff_array:
        """
        Returns the off-diagonal elements of the operator.

        Returns
        -------
        off_diagonals : qcheff_array
            The off-diagonal elements of the operator.
        """

    def __getattr__(self, name) -> Any:
        """
        Forward unknown attributes to the underlying operator.

        Parameters
        ----------
        name : str
            The attribute name.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.op, name)

    def __add__(self, other) -> "OperatorMatrix":
        """
        Element-wise addition of two operators.

        Parameters
        ----------
        other : OperatorMatrix
            The other operator.

        Returns
        -------
        OperatorMatrix
            The sum of the two operators.
        """
        return self.__class__(self.op.__add__(other.op))

    def __sub__(self, other) -> "OperatorMatrix":
        """
        Element-wise subtraction of two operators.

        Parameters
        ----------
        other : OperatorMatrix
            The other operator.

        Returns
        -------
        OperatorMatrix
            The difference of the two operators.
        """
        return self.__class__(self.op.__sub__(other.op))

    def __mul__(self, other: float | complex | int) -> "OperatorMatrix":
        """
        Element-wise multiplication of an operator by a scalar.

        Parameters
        ----------
        other : float | complex | int
            The scalar.

        Returns
        -------
        OperatorMatrix
            The product of the operator and scalar.
        """
        return self.__class__(self.op.__mul__(other))

    def __matmul__(self, other) -> "OperatorMatrix":
        """
        Matrix multiplication of two operators.

        Parameters
        ----------
        other : OperatorMatrix
            The other operator.

        Returns
        -------
        OperatorMatrix
            The matrix product of the two operators.
        """
        return self.__class__(self.op.__matmul__(other.op))

    def __getitem__(self, key) -> qcheff_array:
        """
        Element-wise indexing of an operator.

        Parameters
        ----------
        key : int | slice | tuple
            The indexing key.

        Returns
        -------
        qcheff_array
            The indexed elements of the operator.
        """
        return self.op.__getitem__(key)

    def __len__(self) -> int:
        """
        The number of elements in the operator.

        Returns
        -------
        int
            The number of elements in the operator.
        """
        return len(self.op)

