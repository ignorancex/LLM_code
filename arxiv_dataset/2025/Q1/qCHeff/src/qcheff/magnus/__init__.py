from collections.abc import Sequence

import cupyx.scipy.sparse as cpsparse
import scipy.sparse as spsparse

from qcheff.magnus.magnus import MagnusTimeEvolDense, MagnusTimeEvolSparseLazy
from qcheff.operators.operator_base import OperatorMatrix, qcheff_array


def magnus(
    tlist: qcheff_array,
    drift_ham: OperatorMatrix,
    control_sigs: qcheff_array,
    control_hams: Sequence[OperatorMatrix],
):
    """
    Dispatch Magnus to the appropriate backend based on device of the given array.

    Parameters
    ----------
    tlist : np.ndarray
        Time list.
    drift_ham : np.ndarray or cp.ndarray
        Drift Hamiltonian.
    control_sigs : np.ndarray
        Control signals.
    control_hams : np.ndarray or cp.ndarray
        Control Hamiltonians.

    Returns
    -------
    magnus_class : MagnusTimeEvolDense or MagnusTimeEvolSparseLazy
        An instance of the appropriate Magnus class based on the sparsity of the drift \
        Hamiltonian.
    """

    _is_sparse = cpsparse.issparse(drift_ham.op) or spsparse.issparse(drift_ham.op)
    magnus_class = MagnusTimeEvolSparseLazy if _is_sparse else MagnusTimeEvolDense

    return magnus_class(
        tlist=tlist,
        drift_ham=drift_ham, # type: ignore
        control_sigs=control_sigs,
        control_hams=control_hams, # type: ignore
    )
