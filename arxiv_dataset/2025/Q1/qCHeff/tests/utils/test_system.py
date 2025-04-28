import cupy as cp
import cupyx
import numpy as np
import pytest
import qutip as qt

from qcheff.magnus.magnus import (
    MagnusTimeEvolDense,
    MagnusTimeEvolSparseLazy,
)
from qcheff.utils.pulses import FourierPulse
from qcheff.utils.system import QuTiPSystem


def test_init():
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.sigmay()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    assert system.drift_ham == drift_ham
    assert system.control_sigs == control_sigs
    assert system.control_hams == control_hams


def test_init_error_lengths():
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.sigmay(), qt.sigmaz()]
    with pytest.raises(
        ValueError,
        match="The lengths of control_sigs and control_hams must be the same.",
    ):
        QuTiPSystem(drift_ham, control_sigs, control_hams)


def test_init_error_hermitian():
    drift_ham = qt.Qobj([[1, 2], [3, 4]])
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.sigmay()]
    with pytest.raises(ValueError, match="drift_ham must be Hermitian."):
        QuTiPSystem(drift_ham, control_sigs, control_hams)


def test_init_error_hermitian_control_hams():
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.Qobj([[1, 2], [3, 4]])]
    with pytest.raises(ValueError):
        QuTiPSystem(drift_ham, control_sigs, control_hams)


def test_init_multiple_controls():
    drift_ham = qt.sigmax()
    control_sigs = [
        FourierPulse([1, 2], 1),
        FourierPulse([1, 2], 1),
    ]
    control_hams = [qt.sigmay(), qt.sigmaz()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    assert system.drift_ham == drift_ham
    assert system.control_sigs == control_sigs
    assert system.control_hams == control_hams


def test_get_ham_list():
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]

    control_hams = [qt.sigmay()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    tlist = np.linspace(0, control_sigs[0].gate_time, 100)
    ham_list = system.get_qutip_tdham(tlist)

    assert ham_list[0] == drift_ham
    assert ham_list[1][0] == control_hams[0]
    np.testing.assert_array_almost_equal(ham_list[1][1], control_sigs[0](tlist))


def test_init_no_drift_ham():
    control_sigs = [FourierPulse([1, 2], 1)]

    control_hams = [qt.sigmay()]

    system = QuTiPSystem(None, control_sigs, control_hams)
    tlist = np.linspace(0, control_sigs[0].gate_time, 100)
    ham_list = system.get_qutip_tdham(tlist)

    assert ham_list[0][0] == control_hams[0]
    np.testing.assert_array_almost_equal(ham_list[0][1], control_sigs[0](tlist))


def test_get_ham_list_multiple_controls():
    drift_ham = qt.sigmax()
    control_sigs = [
        FourierPulse([1, 2], 1),
        FourierPulse([1, 2], 1),
    ]
    control_hams = [qt.sigmay(), qt.sigmaz()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    tlist = np.linspace(0, max(sig.gate_time for sig in control_sigs), 100)
    ham_list = system.get_qutip_tdham(tlist)

    assert ham_list[0] == drift_ham
    assert ham_list[1][0] == control_hams[0]
    np.testing.assert_array_almost_equal(ham_list[1][1], control_sigs[0](tlist))
    assert ham_list[2][0] == control_hams[1]
    np.testing.assert_array_almost_equal(ham_list[2][1], control_sigs[1](tlist))


@pytest.mark.parametrize("sparse", ["sparse", "dense"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_get_magnus_system(sparse, device):
    sparse = True if sparse == "sparse" else False
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.sigmay()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)

    tlist = np.linspace(0, 1, 10)
    magnus_system = system.get_magnus_system(tlist, device=device, sparse=sparse)

    if device == "gpu":
        assert (
            "cupy" in cupyx.scipy.get_array_module(magnus_system.drift_ham.op).__name__
        )
    if sparse:
        assert isinstance(magnus_system, MagnusTimeEvolSparseLazy)
    else:
        assert isinstance(magnus_system, MagnusTimeEvolDense)


def test_get_magnus_system_invalid_device():
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.sigmay()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)

    tlist = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        system.get_magnus_system(tlist, device="invalid_device")


def test_get_magnus_system_invalid_sparse():
    drift_ham = qt.sigmax()
    control_sigs = [FourierPulse([1, 2], 1)]
    control_hams = [qt.sigmay()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)

    tlist = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        system.get_magnus_system(tlist, sparse="invalid_sparse")
