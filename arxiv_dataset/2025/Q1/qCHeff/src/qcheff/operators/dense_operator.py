# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field
from types import ModuleType

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import scipy.sparse as spsparse

from qcheff.operators.operator_base import (
    OperatorMatrix,
    qcheff_array,
    qcheff_dense_array,
)


@dataclass
class DenseOperator(OperatorMatrix):
    """
    A dense matrix operator.

    Parameters
    ----------
    op : qcheff_array
        The dense matrix operator.
    backend_module : ModuleType, optional
        The backend module of the operator. Defaults to the module of the input
        operator.

    Attributes
    ----------
    op : qcheff_array
        The dense matrix operator.
    backend_module : ModuleType
        The backend module of the operator.

    """

    op: qcheff_dense_array
    backend_module: ModuleType = field(init=False)

    def __post_init__(
        self,
    ):
        """
        Initialize the operator.

        Notes
        -----
        If the input operator is a sparse matrix, it is converted to a dense
        matrix using the `toarray` method.

        """
        self.backend_module = cp.get_array_module(self.op)
        if cpsparse.issparse(self.op) or spsparse.issparse(self.op):
            self.op = self.op.toarray()

    def save(self, filename: str) -> None:
        """
        Save the operator to a file.

        Parameters
        ----------
        filename : str
            The filename to save to.

        """
        np.savez_compressed(filename, op=self.op)

    @classmethod
    def load(cls, filename: str) -> "DenseOperator":
        """
        Load an operator from a file.

        Parameters
        ----------
        filename : str
            The filename to load from.

        Returns
        -------
        op : DenseOperator
            The loaded operator.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        """
        if not os.path.exists(filename):
            msg = f"File not found: {filename}"
            raise FileNotFoundError(msg)
        data = np.load(filename, allow_pickle=False)
        return cls(op=data["op"])

    def to(self, backend: str) -> None:
        """
        Convert the operator to a different backend.

        Parameters
        ----------
        backend : str
            The backend to convert to.

        Raises
        ------
        ValueError
            If the conversion is not supported.

        """
        if (self.backend_module == np and backend in ["cpu", "numpy", "scipy"]) or (
            self.backend_module == cp and backend in ["gpu", "cupy"]
        ):
            return
        # GPU to CPU conversion
        elif self.backend_module == cp and backend in ["cpu", "numpy", "scipy"]:
            self.op = cp.asnumpy(self.op)

        # CPU to GPU conversion
        elif self.backend_module == np and backend in ["gpu", "cupy"]:
            self.op = cp.array(self.op)
        else:
            msg = f"Unsupported conversion from {self.backend_module} to {backend}."
            raise ValueError(msg)

        # Finally, set the backend module
        self.backend_module = cp.get_array_module(self.op)

    def diagonals(self) -> qcheff_dense_array:
        """
        Return the diagonal elements of the operator.

        Returns
        -------
        diagonals : qcheff_dense_array
            The diagonal elements of the operator.

        """
        return self.op.diagonal()

    def couplings(self) -> qcheff_dense_array:
        """
        Return the upper triangular elements of the operator.

        Returns
        -------
        couplings : qcheff_dense_array
            The upper triangular elements of the operator.

        """
        return np.triu(self.op, k=1)

