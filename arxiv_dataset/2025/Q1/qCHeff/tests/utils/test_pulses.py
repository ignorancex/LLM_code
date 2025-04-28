import numpy as np
import pytest

from qcheff.utils.pulses import ControlPulse, FourierPulse, cos_ramp


def test_cos_ramp():
    assert cos_ramp(0) == 0
    assert cos_ramp(1) == 0
    assert cos_ramp(0.5) == 1


def test_control_pulse_init():
    coeffs = [1, 2]
    basis_funcs = [np.sin, np.cos]
    pulse = ControlPulse(coeffs=coeffs, basis_funcs=basis_funcs)
    assert pulse.coeffs == coeffs
    assert pulse.basis_funcs == basis_funcs
    assert pulse.frequency == 1.0
    assert pulse.phase == 0.0
    assert pulse.amplitude == 1.0


def test_control_pulse_init_error():
    coeffs = [1, 2]
    basis_funcs = [np.sin]  # Different number of basis functions

    with pytest.raises(ValueError):
        ControlPulse(coeffs=coeffs, basis_funcs=basis_funcs)


def test_control_pulse_envelope():
    coeffs = [1, 2]
    basis_funcs = [np.sin, np.cos]
    pulse = ControlPulse(coeffs=coeffs, basis_funcs=basis_funcs)
    tlist = np.linspace(0, 1, 100)
    expected = np.sin(tlist) + 2 * np.cos(tlist)
    np.testing.assert_array_almost_equal(pulse.envelope(tlist), expected)


def test_fourier_pulse_init():
    coeffs = [1, 2]
    gate_time = 1
    pulse = FourierPulse(coeffs=coeffs, gate_time=gate_time)
    assert pulse.coeffs == coeffs
    assert pulse.gate_time == gate_time


def test_fourier_pulse_envelope():
    coeffs = [1, 2]
    gate_time = 1
    pulse = FourierPulse(coeffs=coeffs, gate_time=gate_time)
    tlist = np.linspace(0, 1, 100)
    expected = cos_ramp(tlist) * (np.sin(np.pi * tlist) + 2 * np.cos(2 * np.pi * tlist))
    np.testing.assert_array_almost_equal(pulse.envelope(tlist), expected)


def test_fourier_pulse_str():
    coeffs = [1, 2]
    gate_time = 1
    pulse = FourierPulse(coeffs=coeffs, gate_time=gate_time)
    expected = f"FourierPulseEnvelope(id=None, coeffs={coeffs}, gate_time={gate_time})"
    assert str(pulse) == expected


def test_fourier_pulse_repr():
    coeffs = [1, 2]
    gate_time = 1
    pulse = FourierPulse(coeffs=coeffs, gate_time=gate_time)
    expected = f"FourierPulseEnvelope(id=None, coeffs={coeffs}, gate_time={gate_time})"
    assert repr(pulse) == expected
