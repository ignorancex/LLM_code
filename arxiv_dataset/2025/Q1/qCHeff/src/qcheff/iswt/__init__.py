import cupyx

from qcheff.iswt.iswt import (
    ExactIterativeSWT,
    NPADCupySparse,
    NPADScipySparse,
)
from qcheff.operators import OperatorMatrix, SparseOperator


def NPAD(hamiltonian: OperatorMatrix, **kwargs) -> ExactIterativeSWT:
    """
    Construct an NPAD object from an OperatorMatrix.

    Parameters
    ----------
    hamiltonian : OperatorMatrix
        The operator to be transformed.

    **kwargs
        Additional keyword arguments to be passed to NPADScipySparse or NPADCupySparse,
        depending on the backend of the input operator.

    Returns
    -------
    ExactIterativeSWT
        An ExactIterativeSWT object initialized with the input operator.

    Notes
    -----
    The backend of the input operator is determined using `cupyx.scipy.get_array_module`.
    If the backend is "scipy", NPADScipySparse is used. If the backend is "cupyx.scipy",
    NPADCupySparse is used. Otherwise, a NotImplementedError is raised.
    """
    backend_module = cupyx.scipy.get_array_module(hamiltonian.op).__name__
    # Convert the operator to a SparseOperator for efficiency
    if not isinstance(hamiltonian, SparseOperator):
        hamiltonian = SparseOperator(hamiltonian.op)
    if "cupyx" not in backend_module:
        return NPADScipySparse(H=hamiltonian, **kwargs)
    elif "cupyx" in backend_module:
        return NPADCupySparse(H=hamiltonian, **kwargs)
    else:  # pragma: no cover
        msg = f"Operator type {type(hamiltonian.op)} not supported."
        raise NotImplementedError(msg)
