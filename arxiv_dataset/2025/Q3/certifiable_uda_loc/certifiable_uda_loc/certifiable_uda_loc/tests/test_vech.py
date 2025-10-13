import certifiable_uda_loc.utils.matrix_utils as mat_utils

import numpy as np


def test_vech():
    # Generate random 3x3 matrix
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)

    A = A + A.T
    B = B + B.T
    val1 = np.trace(A @ B)
    val2 = mat_utils.mat2vech(A).reshape(1, -1) @ mat_utils.mat2vech(B).reshape(-1, 1)
    assert np.abs(val1 - val2) < 1e-5

    A_hat = mat_utils.vech2mat(mat_utils.mat2vech(A), A.shape[0])
    assert np.allclose(A, A_hat)


if __name__ == "__main__":
    test_vech()
