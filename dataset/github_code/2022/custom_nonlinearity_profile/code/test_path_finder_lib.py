# Copyright 2022 Salvador Poveda-Hospital

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Basic library for designing waveguides with a given nonlinearity profile along the propagation direction
"""

# pylint: disable=C0103,C0301,R0913

import pytest
import numpy as np
from path_finder_lib import find_path


def analytical(x, x0, y0, b, radius):
    """Gives the analytical solution for a sine nonlinearity with k=1"""
    if b:
        return -np.sqrt(radius**2 - (x - x0) ** 2) + y0
    return np.sqrt(radius**2 - (x - x0) ** 2) + y0


def solution_sine(z, N=30000, Lambda=0.1, k=1):
    """Gives the path for the analytical solution"""
    radius = Lambda * k / 2 / np.pi
    z_test = np.zeros(N)
    x_test = np.zeros(N)
    z_test[0] = 0
    x_test[0] = 0

    for i in range(1, N):
        z_test[i] = z[i - 1]
        if np.mod((z_test[i] - radius), 4 * radius) < 2 * radius:
            b = 0
        else:
            b = 1
        x_test[i] = analytical(
            z_test[i],
            int((z_test[i] + radius) / (2 * radius)) * 2 * radius,
            radius,
            b,
            radius,
        )

    return x_test


@pytest.mark.parametrize("Lambda", [0.1, 0.2, 0.3])
def test_circles(Lambda):
    """Test that the solution from the path finder is correct for a sine nonlinearity"""
    d_sin_local = lambda s: np.sin(2 * np.pi * s / Lambda)
    k = 1
    x, z, _, _ = find_path(d_sin_local, 50000, delta=0.00001, k=k)
    x_test = solution_sine(z, N=50000, Lambda=Lambda, k=1)
    assert np.allclose(x, x_test, atol=0.05)
