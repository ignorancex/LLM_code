import numpy as np

from qcheff.utils.pulses import cos_ramp


def test_cosine_ramp_drive_np():
    t = np.linspace(0, 1, 1000)
    a = 0.2
    ramp = cos_ramp(t, a)

    # Check that the ramp is 0 at t=0 and t=1
    np.testing.assert_almost_equal(ramp[0], 0)
    np.testing.assert_almost_equal(ramp[-1], 0)

    # Check that the ramp is symmetric
    np.testing.assert_array_almost_equal(ramp, ramp[::-1])
