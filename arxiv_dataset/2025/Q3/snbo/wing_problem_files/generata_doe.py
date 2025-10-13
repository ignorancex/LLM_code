from scipy.io import savemat
import numpy as np

def latin_hypercube(n_pts, dim):
    """
        Basic latin hypercube implementation with center perturbation
    """

    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert

    return X

######### Generate DOE

npts = 150
dim = 119

x = latin_hypercube(npts, dim)

######### Save DOE

data = {
    "x": x
}

savemat("doe.mat", data)
