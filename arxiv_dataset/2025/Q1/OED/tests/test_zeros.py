import numpy as np
import pytest
from joblib import delayed, Parallel

from zeros import *


@pytest.fixture
def params():
    """ generate random  m (number of observations) and k (rank of O*O).
    """

    
    params = []
    for m in np.random.randint(low=3, high=80, size=30):
        for k in np.random.randint(low=2, high=m + 1, size=1):
            params.append({'m': m, 'k': k})
            params.append({'m': m, 'k': m})
    return params


def test_theta(big_dim=20, dim=9, upper=4):
    assert big_dim > dim > upper
    M, D, U, S = MDUS(big_dim, dim)

    C = M - np.identity(dim) * np.trace(M) / dim
    assert abs(np.trace(C)) < EPS

    theta, lower = get_theta(C, upper)
    R = givens(theta=theta, dims=dim, lower=lower, upper=upper)
    T = conj(C, R)

    cot = cos(theta) / sin(theta)
    eqn = cot ** 2 * C[upper, upper] + 2 * cot * C[upper, lower] + C[lower, lower]
    assert abs(eqn) < EPS
    assert abs(T[upper, upper]) < EPS
    #print("pass theta")

    
def test_givens(k=8, lower=4, upper=6):
    M, D, U, S = MDUS(k + 2, k)

    ## Testing Givens rotations R
    R = givens(2.3, k, lower, upper)
    C = conj(M, R)

    res = np.abs(M - C)
    res[lower, :] = 0
    res[upper, :] = 0
    res[:, lower] = 0
    res[:, upper] = 0

    assert np.all(np.abs(res) < EPS)
    #$ print("pass givens")

    
def test_all(params):
    Parallel(n_jobs=7)(delayed(caller)(**param) for param in params)
