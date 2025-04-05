import numpy as np
import pytest

from src.forward import Heat
from src.multiplier import FourierMultiplier
from src.observations import DiagObservation, PointObservation
from src.probability import Prior, Posterior

N = 500
L = 3
time = 3e-3
alpha = 0.6
gamma = -1.2
delta = 0.5
TRANSFORMS = ['fft', 'dct']
COLORS = ['r', 'g', 'b', 'k', 'c', 'm', 'y']


@pytest.fixture
def multiplier(transform):
    """Create a multiplier object for testing with eigenvalues
    (10, 9, ..., 1, 0, 0, 0, ...)"""
    multiplier = FourierMultiplier(L=L, N=N, transform=transform)
    multiplier.multiplier = np.zeros(multiplier.N)
    multiplier.multiplier[:10] = np.arange(10, 0, -1)
    return multiplier


@pytest.fixture
def diag_obs(transform):
    obs = DiagObservation(multiplier=1+np.abs(np.random.randn(3)),
                          N=N,
                          L=L,
                          transform=transform)
    return obs


@pytest.fixture
def point_observation(transform):
    meas = np.random.uniform(low=0,high=1, size=8) * L
    return PointObservation(measurements=meas, L=L, N=N, transform=transform)


@pytest.fixture
def many_point(transform):
    meas = np.random.uniform(low=0,high=1, size=120) * L
    return PointObservation(measurements=meas, L=L, N=N, transform=transform)


@pytest.fixture
def prior(transform):
    return Prior(gamma=gamma, delta=delta, N=N, L=L, transform=transform)


@pytest.fixture
def short_prior(transform):
    return Prior(gamma=gamma, delta=delta, N=300, L=L, transform=transform)


@pytest.fixture
def fwd(transform):
    return Heat(N=N, L=L, alpha=alpha, time=time, transform=transform)




@pytest.fixture
def many_point_observation(transform):
    return PointObservation(measurements=np.random.uniform(0, L, 11),
                            L=L,
                            N=10000,
                            transform=transform)


@pytest.fixture
def posterior(transform):
    sig = 1e-2
    forward = Heat(N=N, L=L, transform=transform, alpha=alpha, time=time)
    pr = Prior(N=N, L=L, transform=transform, gamma=gamma, delta=delta)
    return Posterior(fwd=forward,
                     prior=pr,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform)

    
