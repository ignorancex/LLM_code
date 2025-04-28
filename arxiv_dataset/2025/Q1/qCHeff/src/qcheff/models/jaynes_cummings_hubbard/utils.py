# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from itertools import islice, takewhile

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import polars as pl
import qutip as qt
import scipy.sparse as spsparse
from more_itertools import zip_broadcast

from qcheff.iswt.iswt import NPAD
from qcheff.jaynes_cummings_hubbard.models import JCHModel, jch_scqubits_hilbertspace
from qcheff.operators.operator_base import qcheffOperator


@dataclass
class JCHAnalysis:
    """
    NamedTuple containing the Jaynes-Cummings-Hubbard model and the desired level labels.
    """

    model: JCHModel
    level_labels: list
    level_idx: list = field(init=False)
    system_dims: list = field(init=False)

    def __post_init__(self):
        """
        Initializes the level indices for the Jaynes-Cummings model.
        """
        self.system_dims = [2] * self.model.n + [self.model.nr] * self.model.n
        self.level_idx = np.asarray(
            [qt.state_number_index(self.system_dims, x) for x in self.level_labels]
        )

    def analyse(self, methods=None, **kwargs):
        """
        Analyzes the Jaynes-Cummings model and returns the eigenvalues of the system.
        """
        if methods is None:
            methods = ["scqubits", "npad_cpu", "npad_gpu"]
        if "scqubits" in methods:
            yield self.scqubits_eigvals()
        if "npad_cpu" in methods:
            yield self.npad_cpu_eigvals(**kwargs)
        if "npad_gpu" in methods:
            yield self.npad_gpu_eigvals(**kwargs)

    def scqubits_eigvals(self):
        """
        Returns the desired eigenvalues of a Jaynes-Cummings model using SCQubits.

        Energies are returned as a Polars series.
        """

        return self.generate_df(
            evals_list=map(
                jch_scqubits_hilbertspace(model=self.model).energy_by_bare_index,
                self.level_labels,
            ),
            method_name="scqubits",
        )

    def npad_cpu_eigvals(self, batch_size=1, tol=1e-12, max_iter=3):
        """
        Returns the desired eigenvalues of a Jaynes-Cummings model using NPAD.

        Energies are returned as an array of floats.
        """

        test_NPAD = NPAD(
            qcheffOperator(
                spsparse.csr_array(
                    jch_scqubits_hilbertspace(model=self.model).hamiltonian()[:]
                )
            )
        )

        def off_diag_norm(x):
            return spsparse.linalg.norm(x.op - spsparse.diags(x.diagonals()))

        def not_diag(state):
            return off_diag_norm(state[1]) > tol

        for cpl_batch, _ in islice(
            takewhile(
                not_diag,
                test_NPAD.largest_couplings(n=batch_size),
            ),
            max_iter,
        ):
            test_NPAD.eliminate_couplings(cpl_batch)

        return self.generate_df(
            evals_list=np.take(
                a=np.real(test_NPAD.H.diagonals()), indices=self.level_idx
            ),
            method_name="npad_cpu",
        )

    def npad_gpu_eigvals(self, batch_size=1, tol=1e-12, max_iter=3):
        """
        Returns the desired eigenvalues of a Jaynes-Cummings model using NPAD.

        Energies are returned as an array of floats.
        """
        test_NPAD = NPAD(
            qcheffOperator(
                cpsparse.csr_matrix(
                    spsparse.csr_array(
                        jch_scqubits_hilbertspace(model=self.model).hamiltonian()[:]
                    )
                )
            )
        )

        def off_diag_norm(x):
            return cpsparse.linalg.norm(x.op - cpsparse.diags(x.diagonals()))

        def not_diag(state):
            return off_diag_norm(state[1]) > tol

        for cpl_batch, _ in islice(
            takewhile(
                not_diag,
                test_NPAD.largest_couplings(n=batch_size),
            ),
            max_iter,
        ):
            test_NPAD.eliminate_couplings(cpl_batch)

        return self.generate_df(
            evals_list=np.take(
                a=np.real(test_NPAD.H.diagonals()), indices=self.level_idx
            ),
            method_name="npad_gpu",
        )

    def generate_df(self, evals_list, method_name, params=None):
        """
        Generates a DataFrame with the eigenvalues of the Jaynes-Cummings model.
        """
        if params is None:
            params = ["delta", "g", "kappa"]
        return pl.DataFrame(
            data=zip_broadcast(
                method_name,
                # *self.model._asdict().values(),
                *list(map(self.model._asdict().get, params)),
                ("".join(map(str, x)) for x in self.level_labels),
                evals_list,
            ),
            schema=[
                "Method",
                # *self.model._asdict().keys(),
                *params,
                "state",
                "energy",
            ],
        )
