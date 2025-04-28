# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field
from types import ModuleType

import cupyx
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as spsparse

from qcheff.operators.operator_base import (
    OperatorMatrix,
    SparseMatrix,
)


@dataclass
class SparseOperator(OperatorMatrix):
    """
    A sparse matrix operator.

    Attributes
    ----------
    op : SparseMatrix
        The underlying sparse matrix operator.
    backend_module : ModuleType
        The module providing the backend for the operator.
    """

    op: SparseMatrix
    backend_module: ModuleType = field(init=False)

    def __post_init__(self):
        """
        Initialize the operator.

        Notes
        -----
        If the input operator is a dense matrix, it is converted to a sparse
        matrix using the `toarray` method.
        """
        self.backend_module = cupyx.scipy.get_array_module(self.op).sparse
        if not self.backend_module.issparse(self.op):
            # This is only needed due to the API differences between scipy and cupy
            if "cupy" in self.backend_module.__name__:
                self.op = cpsparse.csr_matrix(self.op)
            else:
                self.op = spsparse.csr_array(self.op)

    def save(self, filename: str) -> None:
        """
        Save the OperatorMatrix to disk.

        Parameters
        ----------
        filename : str
            The filename to save to.
        """
        if "cupyx" in self.backend_module.__name__:
            op_to_save = self.op.get()
        else:
            op_to_save = self.op
        spsparse.save_npz(filename, op_to_save)

    @classmethod
    def load(cls, filename: str) -> OperatorMatrix:
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
        if not os.path.exists(filename):
            msg = f"File not found: {filename}"
            raise FileNotFoundError(msg)

        return cls(spsparse.load_npz(filename))

    def to(self, backend: str) -> None:
        """
        Convert the operator to a different backend.
        Note that CPU to GPU conversion incurs memory transfer and is slow.

        Parameters
        ----------
        backend : str
            The backend to convert to.
        """
        _is_gpu_array = "cupyx" in self.backend_module.__name__
        # Same as current backend
        if (_is_gpu_array and backend in ["gpu", "cupy"]) or (
            not _is_gpu_array and backend in ["cpu", "numpy", "scipy"]
        ):
            return

        # GPU to CPU conversion
        elif _is_gpu_array and backend in ["cpu", "scipy", "numpy"]:
            self.op = spsparse.csr_array(self.op.get())

        # CPU to GPU conversion
        elif not _is_gpu_array and backend in ["gpu", "cupy"]:
            self.op = cpsparse.csr_matrix(self.op)
        else:
            msg = f"Unsupported conversion from {self.backend_module.__name__} \
                to {backend}."
            raise ValueError(msg)

        # Finally, set the backend module
        self.backend_module = cupyx.scipy.get_array_module(self.op.data).sparse

    def diagonals(self) -> SparseMatrix:
        """
        Returns the diagonal elements of the operator.

        Returns
        -------
        SparseMatrix
            The diagonal elements of the operator.
        """
        return self.op.diagonal()

    def couplings(self) -> SparseMatrix:
        """
        Returns the off-diagonal elements of the operator.

        Returns
        -------
        SparseMatrix
            The off-diagonal elements of the operator.
        """
        return self.backend_module.triu(self.op, k=1)

