import numpy as np


def rand(d, cov=None):
    if cov is None:
        return np.random.normal(size=d)
    # if it's an array make a diagonal matrix out of it
    if len(cov.shape) == 1:
        return np.random.normal(size=d) * np.sqrt(cov)
    return np.random.multivariate_normal(np.zeros(d), cov)
