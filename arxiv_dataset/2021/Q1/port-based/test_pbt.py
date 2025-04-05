from pbt import GUE0_lambda_max_estimate
import numpy as np

# test GUE0 for d=2
got = GUE0_lambda_max_estimate(2, 100000)
expected = np.sqrt(4 / np.pi)
err = np.abs(got - expected)
assert err < 0.01, err
