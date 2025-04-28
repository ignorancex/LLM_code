# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import cupyx
import cupyx.profiler
import cupyx.scipy.sparse as cpsparse
import more_itertools
import numpy as np
import polars as pl
import qutip as qt
import scipy.sparse as spsparse
from joblib import Memory
from more_itertools import unzip, zip_broadcast

from qcheff.iswt import NPAD, NPADCupySparse, NPADScipySparse
from qcheff.models.jaynes_cummings.models import JCModel
from qcheff.operators import qcheffOperator

cachedir = ".cache"
memory = Memory(cachedir, verbose=0)


def create_bench_df(bench_results, method_name: str, **kwargs):
    incl_params_dict = kwargs.get("incl_params_dict", {})

    return pl.DataFrame(
        more_itertools.zip_broadcast(
            *list(incl_params_dict.values()),
            method_name,
            bench_results.cpu_times,
            bench_results.gpu_times.squeeze(),
        ),
        schema=[*list(incl_params_dict.keys()), "Method", "cpu_time", "gpu_time"],
    )


@memory.cache
def get_sparse_hamiltonian(model: JCModel):
    return qcheffOperator(
        spsparse.csr_array(model.jc_scqubits_hilbertspace().hamiltonian()[:])
    )


def npad_eigvals_func(NPAD, couplings):  # noqa: ARG001
    # NPAD.eliminate_couplings(couplings)
    NPAD.givens_rotation_matrix(0, 1)
    return NPAD.H.diagonal()


def scqubits_eigvals_func(ham):
    return ham.eigenenergies()


@dataclass
class JCMottAnalysis:
    model: JCModel
    level_labels: list

    def benchmark(self, **kwargs):
        """
        Benchmarks the Mott Analysis using the specified methods.
        """
        test_ham = self.model.jc_scqubits_hilbertspace().hamiltonian()[:]

        test_NPAD_cupy = NPADCupySparse(
            qcheffOperator(cpsparse.csr_matrix(spsparse.csr_array(test_ham)))
        )

        test_NPAD_scipy = NPADScipySparse(qcheffOperator(spsparse.csr_array(test_ham)))

        methods = kwargs.get("methods", ["scQubits", "NPAD (CPU)", "NPAD (GPU)"])
        num_couplings = kwargs.get("num_couplings", 50)
        couplings = [
            (i, (i - 1) + self.model.resonator_levels) for i in range(1, num_couplings)
        ]

        return pl.concat(
            items=(
                create_bench_df(
                    cupyx.profiler.benchmark(
                        callable_func,
                        args=func_args,
                        n_repeat=10,
                        n_warmup=2,
                    ),
                    method_name=method_name,
                    **kwargs,
                )
                for method_name, callable_func, func_args in zip(
                    ["scQubits", "NPAD (CPU)", "NPAD (GPU)"],
                    (
                        scqubits_eigvals_func,
                        npad_eigvals_func,
                        npad_eigvals_func,
                    ),
                    (
                        (qt.Qobj(test_ham).to("csr"),),
                        (test_NPAD_scipy, couplings),
                        (test_NPAD_cupy, couplings),
                    ),
                    strict=True,
                )
                if method_name in methods
            )
        )

    def analyse(self, methods=None, detuning_list=None, **kwargs):
        """
        Analyzes the Jaynes-Cummings model and returns the eigenvalues of the system.
        """
        if methods is None:
            methods = ["scqubits", "npad_cpu", "npad_gpu"]

        if detuning_list is None:
            # Only do the analysis for a single detuning
            detuning_list = [self.model.detuning]

        for detuning in detuning_list:
            self.model.detuning = detuning
            for method in methods:
                yield getattr(self, f"{method}_eigvals")(**kwargs)

    def scqubits_eigvals(self):
        """
        Returns the desired eigenvalues of a Jaynes-Cummings model using SCQubits.

        Energies are returned as a Polars series.
        """
        return pl.DataFrame(
            data=zip_broadcast(
                "scQubits",
                *self.model.__dict__.values(),
                *unzip(self.level_labels),
                map(
                    self.model.jc_scqubits_hilbertspace().energy_by_bare_index,
                    self.level_labels,
                ),
            ),
            schema=[
                "Method",
                *self.model.__dict__.keys(),
                "qubit state",
                "resonator state",
                "energy",
            ],
        )

    def npad_cpu_eigvals(self):
        """
        Returns the desired eigenvalues of a Jaynes-Cummings model using NPAD.

        Energies are returned as an array of floats.
        """

        test_NPAD = NPAD(
            qcheffOperator(
                spsparse.csr_array(
                    self.model.jc_scqubits_hilbertspace().hamiltonian()[:]
                )
            )
        )

        test_NPAD.eliminate_couplings(
            [
                (i, i - 1 + self.model.resonator_levels)
                for i in range(1, len(self.level_labels) - 1)
            ]
        )

        return pl.DataFrame(
            data=zip_broadcast(
                "NPAD (CPU)",
                *self.model.__dict__.values(),
                *unzip(self.level_labels),
                np.take(
                    np.real(test_NPAD.H.diagonals()),
                    indices=[
                        qt.state_number_index([2, self.model.resonator_levels], label)
                        for label in self.level_labels
                    ],
                ),
            ),
            schema=[
                "Method",
                *self.model.__dict__.keys(),
                "qubit state",
                "resonator state",
                "energy",
            ],
        )

    def npad_gpu_eigvals(self):
        """
        Returns the desired eigenvalues of a

        Jaynes-Cummings model using NPAD.
        """
        """
        Returns the desired eigenvalues of a Jaynes-Cummings model using NPAD.

        Energies are returned as an array of floats.
        """
        test_NPAD = NPAD(
            qcheffOperator(
                cpsparse.csr_matrix(
                    spsparse.csr_array(
                        self.model.jc_scqubits_hilbertspace().hamiltonian()[:]
                    )
                )
            )
        )

        test_NPAD.eliminate_couplings(
            [
                (i, (i - 1) + self.model.resonator_levels)
                for i in range(1, len(self.level_labels) - 1)
            ]
        )

        return pl.DataFrame(
            data=zip_broadcast(
                "NPAD (GPU)",
                *self.model.__dict__.values(),
                *unzip(self.level_labels),
                np.take(
                    np.real(test_NPAD.H.diagonals()),
                    indices=[
                        qt.state_number_index([2, self.model.resonator_levels], label)
                        for label in self.level_labels
                    ],
                ),
            ),
            schema=[
                "Method",
                *self.model.__dict__.keys(),
                "qubit state",
                "resonator state",
                "energy",
            ],
        )
